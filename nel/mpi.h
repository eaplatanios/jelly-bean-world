#ifndef NEL_MPI_H_
#define NEL_MPI_H_

/* BIG TODO: test this on Windows */

#include <core/array.h>
#include <stdio.h>
#include <thread>
#include <condition_variable>

#if defined(_WIN32)
#include <winsock2.h>
typedef SOCKET socket_type;
typedef int socklen_t;
#else
#include <sys/socket.h>
#include <netdb.h>
typedef int socket_type;
#endif

inline bool valid_socket(const socket_type& sock) {
#if defined(_WIN32)
	return sock != INVALID_SOCKET;
#else
	return sock >= 0;
#endif
}

inline void mpi_error(const char* message) {
#if defined(_WIN32)
	errno = WSAGetLastError();
#endif
	perror(message);
}

template<typename ProcessConnectionCallback>
void run_worker(array<socket_type>& connection_queue,
		std::condition_variable& cv, std::mutex& lock,
		ProcessConnectionCallback process_connection,
		bool& server_running)
{
	while (server_running) {
		std::unique_lock<std::mutex> lck(lock);
		cv.wait(lck);
		socket_type connection = connection_queue.pop();
		lck.unlock();

		process_connection(connection);
		close(connection);
	}
}

template<typename ProcessConnectionCallback>
bool run_server(
		const char* server_address, const char* server_port,
		unsigned int connection_queue_capacity, unsigned int worker_count,
		ProcessConnectionCallback process_connection, bool& server_running)
{
	addrinfo hints;
	hints.ai_family = AF_UNSPEC;
	hints.ai_socktype = SOCK_STREAM;
	hints.ai_protocol = IPPROTO_TCP;

	addrinfo* addresses;
	int result = get_addr_info(server_address, server_port, &hints, &addresses);
	if (result != 0 || addresses == NULL) {
		fprintf(stderr, "run_client ERROR: Unable to resolve address. %s\n", gai_strerror(result));
		return false;
	}

	socket_type sock;
	for (addrinfo* entry = addresses; entry != NULL; entry++) {
		sock = socket(entry->ai_family, entry->ai_socktype, entry->ai_protocol);
		if (!valid_socket(sock)) {
			mpi_error("run_server ERROR: Unable to open socket. ");
			continue;
		}

		int yes = 1;
		if (setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes)) != 0) {
			mpi_error("run_server ERROR: Unable to set socket option. ");
			return false;
		}

		if (!bind_socket(sock, entry->ai_addr, entry->ai_addrlen)) {
			close(sock);
			continue;
		}
		break;
	}
	freeaddrinfo(addresses);

	if (listen(sock, connection_queue_capacity) != 0) {
		mpi_error("run_server ERROR: Unable to listen to socket. ");
		close(sock); return false;
	}

	/* make the thread pool */
	std::condition_variable cv; std::mutex lock;
	array<socket_type> connection_queue(64);
	std::thread* workers = new std::thread[worker_count];
	auto dispatch = [&]() {
		run_worker(connection_queue, cv, lock, process_connection, server_running);
	};
	for (unsigned int i = 0; i < worker_count; i++)
		threads[i] = std::thread(dispatch);

	/* the main loop */
	sockaddr_storage client_address;
	socklen_t address_size = sizeof(client_address);
	while (server_running) {
		socket_type connection = accept(sock, (sockaddr*) &client_address, &address_size);
		if (!valid_socket(connection)) {
			mpi_error("run_server ERROR: Error establishing connection with client. ");
			continue;
		}

		connection_queue.add(connection);
		cv.notify_one();
	}

	for (unsigned int i = 0; i < worker_count; i++)
		threads[i].join();
	delete[] threads;
	close(sock);
	return true;
}

template<typename ProcessConnectionCallback>
bool run_client(client& new_client,
		const char* server_address, const char* server_port,
		ProcessConnectionCallback process_connection)
{
	addrinfo hints;
	hints.ai_family = AF_UNSPEC;
	hints.ai_socktype = SOCK_STREAM;

	addrinfo* addresses;
	int result = get_addr_info(server_address, server_port, &hints, &addresses);
	if (result != 0 || addresses == NULL) {
		fprintf(stderr, "run_client ERROR: Unable to resolve address. %s\n", gai_strerror(result));
		return false;
	}

	socket_type sock;
	for (addrinfo* entry = addresses; entry != NULL; entry++) {
		sock = socket(entry->ai_family, entry->ai_socktype, entry->ai_protocol);
		if (!valid_socket(sock)) {
			mpi_error("run_client ERROR: Unable to open socket. ");
			continue;
		}

		if (connect(sock, p->ai_addr, p->ai_addrlen) != 0) {
			mpi_error("run_client ERROR: Unable to connect. ");
			close(sock); continue;
		}
		break;
	}
	freeaddrinfo(addresses);

	ProcessConnection(sock);
	return true;
}

enum class message_type : uint64_t {
	MOVE = 0,
	MOVE_RESPONSE = 1
};

typedef void (*response_callback)();

struct async_server {
	std::thread server_thread;
	bool server_running;
};

bool init_server(async_server& new_server,
		const char* server_address, const char* server_port,
		unsigned int connection_queue_capacity, unsigned int worker_count)
{
	auto dispatch = [&]() {
		run_server(server_address, server_port, connection_queue_capacity, worker_count, process_server_connection);
	};
	new_server.server_running = true;
	new_server.server_thread = std::thread(dispatch);
}

inline bool init_server(
		const char* server_address, const char* server_port,
		unsigned int connection_queue_capacity, unsigned int worker_count)
{
	run_server(server_address, server_port, connection_queue_capacity, worker_count, process_server_connection);
}

void stop_server(async_server& server) {
	server.server_running = false;
	server.server_thread.join();
}


struct client {
	socket_type connection;
	std::thread response_listener;
	response_callback on_move_response;
};

bool send_move(client& c, direction dir, unsigned int num_steps) {
	memory_stream out = memory_stream(sizeof(message_type) + sizeof(direction) + sizeof(num_steps));
	if (!write(message_type::MOVE, out)
	 || !write(dir, out)
	 || !write(num_steps, out)
	 || send(c.connection, out.buffer, out.length, 0) != 0)
		return false;
	return true;
}

void run_response_listener(client& c) {
	while (c.alive) {
		message_type type;
		read(c.connection, type);
		switch (type) {
			case MOVE_RESPONSE:
				c.on_response();
			default:
				fprintf(stderr, "run_response_listener ERROR: Received invalid message type from server.\n");
				continue;
		}
	}
}

bool init_client(client& new_client, response_callback 
		const char* server_address, const char* server_port)
{
	auto process_connection = [&](socket_type& connection) {
		auto dispatch = [&]() {
			run_response_listener(new_client);
		};
		new_client.connection = connection;
		new_client.response_listener = std::thread(dispatch);
	};

	return run_client(server_address, server_port, process_connection);
}

#endif /* NEL_MPI_H_ */
