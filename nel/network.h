#ifndef NEL_NETWORK_H_
#define NEL_NETWORK_H_

/* BIG TODO: test this on Windows */

#include <core/array.h>
#include <core/map.h>
#include <stdio.h>
#include <thread>
#include <condition_variable>

#if defined(_WIN32) /* on Windows */

#include <winsock2.h>
#include <wepoll.h>
typedef SOCKET socket_type;
typedef int socklen_t;
typedef HANDLE epoll_type;

#else /* on Mac and Linux */

#include <sys/epoll.h>
#include <sys/socket.h>
#include <netdb.h>
#include <unistd.h>
typedef int socket_type;
typedef int epoll_type;

inline void epoll_close(epoll_type& instance) {
	close(instance);
}

#endif


#define EVENT_QUEUE_CAPACITY 1024


namespace nel {

using namespace core;

/**
 * Reads `sizeof(T)` bytes from `in` and writes them to the memory referenced
 * by `value`. This function does not perform endianness transformations.
 * \param in a handle to a socket.
 * \tparam T satisfies [is_fundamental](http://en.cppreference.com/w/cpp/types/is_fundamental).
 */
template<typename T, typename std::enable_if<std::is_fundamental<T>::value>::type* = nullptr>
inline bool read(T& value, socket_type& in) {
	return (recv(in, &value, sizeof(T), MSG_WAITALL) > 0);
}

/**
 * Writes `sizeof(T)` bytes to `out` from the memory referenced by `value`.
 * This function does not perform endianness transformations.
 * \param out a handle to a socket.
 * \tparam T satisfies [is_fundamental](http://en.cppreference.com/w/cpp/types/is_fundamental).
 */
template<typename T, typename std::enable_if<std::is_fundamental<T>::value>::type* = nullptr>
inline bool write(const T& value, socket_type& out) {
	return (send(out, &value, sizeof(T), 0) > 0);
}

inline bool valid_socket(const socket_type& sock) {
#if defined(_WIN32)
	return sock != INVALID_SOCKET;
#else
	return sock >= 0;
#endif
}

inline bool valid_epoll(const epoll_type& instance) {
#if defined(_WIN32)
	return instance != NULL;
#else
	return instance != -1;
#endif
}

inline void network_error(const char* message) {
#if defined(_WIN32)
	errno = WSAGetLastError();
#endif
	perror(message);
}

template<typename ProcessMessageCallback, typename... CallbackArgs>
void run_worker(array<socket_type>& connection_queue,
		std::condition_variable& cv, std::mutex& lock, bool& server_running,
		ProcessMessageCallback process_message, CallbackArgs&&... callback_args)
{
	while (server_running) {
		std::unique_lock<std::mutex> lck(lock);
		cv.wait(lck);
		socket_type connection = connection_queue.pop();
		lck.unlock();

		process_message(connection, std::forward<CallbackArgs>(callback_args)...);
fprintf(stderr, "run_worker: Finished process_message.\n");
	}
}

template<typename ProcessMessageCallback, typename... CallbackArgs>
bool run_server(socket_type& sock, uint16_t server_port,
		unsigned int connection_queue_capacity, unsigned int worker_count,
		bool& server_running, std::condition_variable& init_cv,
		ProcessMessageCallback process_message, CallbackArgs&&... callback_args)
{
	sock = socket(AF_INET6, SOCK_STREAM, 0);
	if (!valid_socket(sock)) {
		network_error("run_server ERROR: Unable to open socket");
		shutdown(sock, 2); server_running = false; init_cv.notify_all();
		return false;
	}

	int yes = 1;
	if (setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes)) != 0) {
		network_error("run_server ERROR: Unable to set socket option");
		server_running = false; init_cv.notify_all(); return false;
	}

	sockaddr_in6 server_addr;
	memset(&server_addr, 0, sizeof(server_addr));
	server_addr.sin6_family = AF_INET6;
	server_addr.sin6_port = htons(server_port);
	server_addr.sin6_addr = in6addr_any;
	if (bind(sock, (sockaddr*) &server_addr, sizeof(server_addr)) != 0) {
		network_error("run_server ERROR: Unable to bind to socket");
		shutdown(sock, 2); server_running = false; init_cv.notify_all();
		return false;
	}

	if (listen(sock, connection_queue_capacity) != 0) {
		network_error("run_server ERROR: Unable to listen to socket");
		shutdown(sock, 2); server_running = false; init_cv.notify_all();
		return false;
	}

	epoll_type epoll_instance = epoll_create1(0);
	if (!valid_epoll(epoll_instance)) {
		network_error("run_server ERROR: Failed to initialize network event polling");
		shutdown(sock, 2); server_running = false; init_cv.notify_all();
		return false;
	}

	epoll_event events[EVENT_QUEUE_CAPACITY];
	epoll_event new_event;
	new_event.events = EPOLLIN;
	new_event.data.fd = sock;
	if (epoll_ctl(epoll_instance, EPOLL_CTL_ADD, sock, &new_event) == -1) {
		network_error("run_server ERROR: Failed to add network polling event");
		shutdown(sock, 2); epoll_close(epoll_instance);
		server_running = false; init_cv.notify_all();
		return false;
	}

	/* make the thread pool */
	std::condition_variable cv; std::mutex lock;
	hash_set<socket_type> connections(1024);
	array<socket_type> connection_queue(64);
	std::thread* workers = new std::thread[worker_count];
	auto start_worker = [&]() {
		run_worker(connection_queue, cv, lock, server_running, process_message, std::forward<CallbackArgs>(callback_args)...); };
	for (unsigned int i = 0; i < worker_count; i++)
		workers[i] = std::thread(start_worker);

	/* notify that the server has successfully started */
	init_cv.notify_all();

	/* the main loop */
	sockaddr_storage client_address;
	socklen_t address_size = sizeof(client_address);
	while (server_running) {
		int event_count = epoll_wait(epoll_instance, events, EVENT_QUEUE_CAPACITY, -1);
		if (event_count == -1) {
			network_error("run_server ERROR: Error waiting for incoming network activity");
			continue;
		}

		for (unsigned int i = 0; i < event_count; i++) {
			if (events[i].data.fd == sock) {
				/* there's a new connection */
				socket_type connection = accept(sock, (sockaddr*) &client_address, &address_size);
				if (!valid_socket(connection)) {
					network_error("run_server ERROR: Error establishing connection with client");
					continue;
				}

				epoll_event new_event;
				new_event.events = EPOLLIN;
				new_event.data.fd = connection;
				if (epoll_ctl(epoll_instance, EPOLL_CTL_ADD, connection, &new_event) == -1) {
					network_error("run_server ERROR: Failed to add network polling event");
					shutdown(connection, 2); continue;
				}
				connections.add(connection);
			} else {
				if (events[i].events & EPOLLIN) {
					/* there is incoming data from a client */
					lock.lock();
					connection_queue.add(events[i].data.fd);
					lock.unlock(); cv.notify_one();
				} if (events[i].events & (EPOLLHUP | EPOLLERR)) {
					/* the client has closed this connection, so remove it */
					if (epoll_ctl(epoll_instance, EPOLL_CTL_DEL, events[i].data.fd, NULL) == -1) {
						network_error("run_server ERROR: Failed to remove network polling event");
						shutdown(events[i].data.fd, 2); continue;
					}
					connections.remove(events[i].data.fd);
					shutdown(events[i].data.fd, 2);
				}
			}
		}
	}

	for (socket_type& connection : connections)
		shutdown(connection, 2);
	shutdown(sock, 2);
	epoll_close(epoll_instance);
	for (unsigned int i = 0; i < worker_count; i++)
		workers[i].join();
	delete[] workers;
	return true;
}

template<typename ProcessConnectionCallback>
bool run_client(
		const char* server_address, const char* server_port,
		ProcessConnectionCallback process_connection)
{
	addrinfo hints;
	memset(&hints, 0, sizeof(hints));
	hints.ai_family = AF_UNSPEC;
	hints.ai_socktype = SOCK_STREAM;
	hints.ai_protocol = IPPROTO_TCP;

	addrinfo* addresses;
	int result = getaddrinfo(server_address, server_port, &hints, &addresses);
	if (result != 0 || addresses == NULL) {
		fprintf(stderr, "run_client ERROR: Unable to resolve address. %s\n", gai_strerror(result));
		return false;
	}

	socket_type sock;
	for (addrinfo* entry = addresses; entry != NULL; entry++) {
		sock = socket(entry->ai_family, entry->ai_socktype, entry->ai_protocol);
		if (!valid_socket(sock)) {
			network_error("run_client ERROR: Unable to open socket");
			continue;
		}

		if (connect(sock, entry->ai_addr, entry->ai_addrlen) != 0) {
			network_error("run_client ERROR: Unable to connect");
			shutdown(sock, 2); continue;
		}
		break;
	}
	freeaddrinfo(addresses);

	process_connection(sock);
	return true;
}

} /* namespace nel */

#endif /* NEL_NETWORK_H_ */
