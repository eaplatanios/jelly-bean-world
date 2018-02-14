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
typedef int socklen_t;
typedef HANDLE epoll_type;

#else /* on Mac and Linux */

#include <sys/epoll.h>
#include <sys/socket.h>
#include <netdb.h>
#include <unistd.h>
typedef int epoll_type;

inline void epoll_close(epoll_type& instance) {
	close(instance);
}

#endif


#define EVENT_QUEUE_CAPACITY 1024


namespace nel {

using namespace core;

struct socket_type {
#if defined(_WIN32)
	SOCKET handle;

	static constexpr SOCKET EMPTY_SOCKET = INVALID_SOCKET;

	socket_type() { }
	socket_type(const SOCKET& src) : handle(src) { }
#else
	int handle;

	static constexpr int EMPTY_SOCKET = -1;

	socket_type() { }
	socket_type(int src) : handle(src) { }
#endif

	inline bool operator == (const socket_type& other) const {
		return handle == other.handle;
	}

	static inline bool is_empty(const socket_type& key) {
		return key.handle == EMPTY_SOCKET;
	}

	static inline void set_empty(socket_type& key) {
		key.handle = EMPTY_SOCKET;
	}

	static inline unsigned int hash(const socket_type& key) {
		return default_hash(key.handle);
	}

	static inline void move(const socket_type& src, socket_type& dst) {
		dst.handle = src.handle;
	}
};

void* alloc_socket_keys(size_t n, size_t element_size) {
	socket_type* keys = (socket_type*) malloc(sizeof(socket_type) * n);
	if (keys == NULL) return NULL;
	for (unsigned int i = 0; i < n; i++)
		socket_type::set_empty(keys[i]);
	return (void*) keys;
}

/**
 * Reads `sizeof(T)` bytes from `in` and writes them to the memory referenced
 * by `value`. This function does not perform endianness transformations.
 * \param in a handle to a socket.
 * \tparam T satisfies [is_fundamental](http://en.cppreference.com/w/cpp/types/is_fundamental).
 */
template<typename T, typename std::enable_if<std::is_fundamental<T>::value>::type* = nullptr>
inline bool read(T& value, socket_type& in) {
	return (recv(in.handle, &value, sizeof(T), MSG_WAITALL) > 0);
}

/**
 * Reads `length` elements from `in` and writes them to the native array
 * `values`. This function does not perform endianness transformations.
 * \param in a handle to a socket.
 * \tparam T satisfies [is_fundamental](http://en.cppreference.com/w/cpp/types/is_fundamental).
 */
template<typename T, typename std::enable_if<std::is_fundamental<T>::value>::type* = nullptr>
inline bool read(T* values, socket_type& in, unsigned int length) {
	return (recv(in.handle, values, sizeof(T) * length, MSG_WAITALL) > 0);
}

/**
 * Writes `sizeof(T)` bytes to `out` from the memory referenced by `value`.
 * This function does not perform endianness transformations.
 * \param out a handle to a socket.
 * \tparam T satisfies [is_fundamental](http://en.cppreference.com/w/cpp/types/is_fundamental).
 */
template<typename T, typename std::enable_if<std::is_fundamental<T>::value>::type* = nullptr>
inline bool write(const T& value, socket_type& out) {
	return (send(out.handle, &value, sizeof(T), 0) > 0);
}

/**
 * Writes `length` elements to `out` from the native array `values`. This
 * function does not perform endianness transformations.
 * \param out a handle to a socket.
 * \tparam T satisfies [is_fundamental](http://en.cppreference.com/w/cpp/types/is_fundamental).
 */
template<typename T, typename std::enable_if<std::is_fundamental<T>::value>::type* = nullptr>
inline bool write(const T* values, socket_type& out, unsigned int length) {
	return (send(out.handle, values, sizeof(T) * length, 0) > 0);
}

inline void network_error(const char* message) {
#if defined(_WIN32)
	errno = WSAGetLastError();
#endif
	perror(message);
}

inline bool valid_socket(const socket_type& sock) {
#if defined(_WIN32)
	return sock.handle != INVALID_SOCKET;
#else
	return sock.handle >= 0;
#endif
}

inline bool valid_epoll(const epoll_type& instance) {
#if defined(_WIN32)
	return instance != NULL;
#else
	return instance != -1;
#endif
}

struct socket_event {
	socket_type connection;
	uint32_t event_type;
};

template<typename ProcessMessageCallback, typename... CallbackArgs>
void run_worker(
		array<socket_event>& event_queue, epoll_type& epoll_instance, hash_set<socket_type>& connections,
		std::condition_variable& cv, std::mutex& event_queue_lock, std::mutex& connection_set_lock,
		bool& server_running, ProcessMessageCallback process_message, CallbackArgs&&... callback_args)
{
	while (server_running) {
		std::unique_lock<std::mutex> lck(event_queue_lock);
		cv.wait(lck);
		socket_event event = event_queue.pop();
		lck.unlock();

		uint8_t next;
		if (recv(event.connection.handle, &next, sizeof(next), MSG_PEEK) == 0) {
			/* the other end of the socket was closed by the client */
			if (epoll_ctl(epoll_instance, EPOLL_CTL_DEL, event.connection.handle, NULL) == -1)
				network_error("run_worker ERROR: Failed to remove network polling event");
			connection_set_lock.lock();
			connections.remove(event.connection);
			connection_set_lock.unlock();
			shutdown(event.connection.handle, 2);
		} else {
			/* there is a data waiting to be read, so read it */
			process_message(event.connection, std::forward<CallbackArgs>(callback_args)...);

			/* tell epoll to continue polling this socket */
			epoll_event new_event = {0};
			new_event.events = EPOLLONESHOT | EPOLLIN | EPOLLRDHUP | EPOLLHUP | EPOLLERR;
			new_event.data.fd = event.connection.handle;
			if (epoll_ctl(epoll_instance, EPOLL_CTL_MOD, event.connection.handle, &new_event) == -1) {
				network_error("run_worker ERROR: Failed to modify network polling event");
				shutdown(event.connection.handle, 2);
			}
		}
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
		shutdown(sock.handle, 2); server_running = false; init_cv.notify_all();
		return false;
	}

	int yes = 1;
	if (setsockopt(sock.handle, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes)) != 0) {
		network_error("run_server ERROR: Unable to set socket option");
		server_running = false; init_cv.notify_all(); return false;
	}

	sockaddr_in6 server_addr;
	memset(&server_addr, 0, sizeof(server_addr));
	server_addr.sin6_family = AF_INET6;
	server_addr.sin6_port = htons(server_port);
	server_addr.sin6_addr = in6addr_any;
	if (bind(sock.handle, (sockaddr*) &server_addr, sizeof(server_addr)) != 0) {
		network_error("run_server ERROR: Unable to bind to socket");
		shutdown(sock.handle, 2); server_running = false; init_cv.notify_all();
		return false;
	}

	if (listen(sock.handle, connection_queue_capacity) != 0) {
		network_error("run_server ERROR: Unable to listen to socket");
		shutdown(sock.handle, 2); server_running = false; init_cv.notify_all();
		return false;
	}

	epoll_type epoll_instance = epoll_create1(0);
	if (!valid_epoll(epoll_instance)) {
		network_error("run_server ERROR: Failed to initialize network event polling");
		shutdown(sock.handle, 2); server_running = false; init_cv.notify_all();
		return false;
	}

	epoll_event events[EVENT_QUEUE_CAPACITY];
	epoll_event new_event = {0};
	new_event.events = EPOLLIN | EPOLLERR;
	new_event.data.fd = sock.handle;
	if (epoll_ctl(epoll_instance, EPOLL_CTL_ADD, sock.handle, &new_event) == -1) {
		network_error("run_server ERROR: Failed to add network polling event");
		shutdown(sock.handle, 2); epoll_close(epoll_instance);
		server_running = false; init_cv.notify_all();
		return false;
	}

	/* make the thread pool */
	std::condition_variable cv;
	std::mutex event_queue_lock, connection_set_lock;
	hash_set<socket_type> connections(1024, alloc_socket_keys);
	array<socket_event> event_queue(64);
	std::thread* workers = new std::thread[worker_count];
	auto start_worker = [&]() {
		run_worker(event_queue, epoll_instance, connections, cv, event_queue_lock, connection_set_lock,
				server_running, process_message, std::forward<CallbackArgs>(callback_args)...);
	};
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

		for (int i = 0; i < event_count; i++) {
			if (events[i].data.fd == sock.handle) {
				/* there's a new connection */
				socket_type connection = accept(sock.handle, (sockaddr*) &client_address, &address_size);
				if (!valid_socket(connection)) {
					network_error("run_server ERROR: Error establishing connection with client");
					continue;
				}

				epoll_event new_event = {0};
				new_event.events = EPOLLONESHOT | EPOLLIN | EPOLLERR;
				new_event.data.fd = connection.handle;
				if (epoll_ctl(epoll_instance, EPOLL_CTL_ADD, connection.handle, &new_event) == -1) {
					network_error("run_server ERROR: Failed to add network polling event");
					shutdown(connection.handle, 2); continue;
				}
				connection_set_lock.lock();
				connections.add(connection, alloc_socket_keys);
				connection_set_lock.unlock();
			} else {
				/* there is an event on a client connection */
				event_queue_lock.lock();
				event_queue.add({events[i].data.fd, events[i].events});
				event_queue_lock.unlock(); cv.notify_one();
			}
		}
	}

	for (unsigned int i = 0; i < worker_count; i++)
		workers[i].join();
	delete[] workers;

	for (socket_type& connection : connections)
		shutdown(connection.handle, 2);
	shutdown(sock.handle, 2);
	epoll_close(epoll_instance);
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

		if (connect(sock.handle, entry->ai_addr, entry->ai_addrlen) != 0) {
			network_error("run_client ERROR: Unable to connect");
			shutdown(sock.handle, 2); continue;
		}
		break;
	}
	freeaddrinfo(addresses);

	process_connection(sock);
	return true;
}

} /* namespace nel */

#endif /* NEL_NETWORK_H_ */
