#ifndef NEL_MPI_H_
#define NEL_MPI_H_

/* BIG TODO: test this on Windows */

#include <core/array.h>
#include <stdio.h>
#include <thread>
#include <condition_variable>

#include "simulator.h"

#if defined(_WIN32)
#include <winsock2.h>
#include <wepoll.h>
typedef SOCKET socket_type;
typedef int socklen_t;
typedef HANDLE epoll_type;
#else
#include <sys/epoll.h>
#include <sys/socket.h>
#include <netdb.h>
typedef int socket_type;
typedef int epoll_type;
#endif

#define EVENT_QUEUE_CAPACITY 1024

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

inline void mpi_error(const char* message) {
#if defined(_WIN32)
	errno = WSAGetLastError();
#endif
	perror(message);
}

template<typename ProcessMessageCallback>
void run_worker(array<socket_type>& connection_queue,
		std::condition_variable& cv, std::mutex& lock,
		ProcessMessageCallback process_message,
		simulator& sim, bool& server_running)
{
	while (server_running) {
		std::unique_lock<std::mutex> lck(lock);
		cv.wait(lck);
		socket_type connection = connection_queue.pop();
		lck.unlock();

		process_message(connection, sim);
	}
}

template<typename ProcessMessageCallback>
bool run_server(
		const char* server_address, const char* server_port,
		unsigned int connection_queue_capacity, unsigned int worker_count,
		ProcessMessageCallback process_message, simulator& sim,
		bool& server_running, std::condition_variable& init_cv)
{
	addrinfo hints;
	hints.ai_family = AF_UNSPEC;
	hints.ai_socktype = SOCK_STREAM;
	hints.ai_protocol = IPPROTO_TCP;

	addrinfo* addresses;
	int result = get_addr_info(server_address, server_port, &hints, &addresses);
	if (result != 0 || addresses == NULL) {
		fprintf(stderr, "run_client ERROR: Unable to resolve address. %s\n", gai_strerror(result));
		server_running = false; init_cv.notify_one(); return false;
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
			server_running = false; init_cv.notify_one(); return false;
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
		close(sock); server_running = false; init_cv.notify_one();
		return false;
	}

	epoll_type epoll_instance = epoll_create(0);
	if (valid_epoll(epoll_instance)) {
		mpi_error("run_server ERROR: Failed to initialize network event polling. ");
		close(sock); server_running = false; init_cv.notify_one();
		return false;
	}

	epoll_event events[EVENT_QUEUE_CAPACITY];
	epoll_event new_event;
	new_event.events = EPOLLIN;
	new_event.data.fd = sock;
	if (epoll_ctl(epoll_instance, EPOLL_CTL_ADD, sock, &new_event) == -1) {
		mpi_error("run_server ERROR: Failed to add network polling event. ");
		close(sock); epoll_close(epoll_instance);
		server_running = false; init_cv.notify_one();
		return false;
	}

	/* make the thread pool */
	std::condition_variable cv; std::mutex lock;
	array<socket_type> connection_queue(64);
	std::thread* workers = new std::thread[worker_count];
	auto start_worker = [&]() {
		run_worker(connection_queue, cv, lock, process_connection, sim, server_running); };
	for (unsigned int i = 0; i < worker_count; i++)
		threads[i] = std::thread(start_worker);

	/* notify that the server has successfully started */
	init_cv.notify_one();

	/* the main loop */
	sockaddr_storage client_address;
	socklen_t address_size = sizeof(client_address);
	while (server_running) {
		int event_count = epoll_wait(epoll_instance, events, EVENT_QUEUE_CAPACITY, -1);
		if (event_count == -1) {
			mpi_error("run_server ERROR: Error waiting for incoming network activity. ");
			continue;
		}

		for (unsigned int i = 0; i < event_count; i++) {
			if (events[i].data.fd == sock) {
				/* there's a new connection */
				socket_type connection = accept(sock, (sockaddr*) &client_address, &address_size);
				if (!valid_socket(connection)) {
					mpi_error("run_server ERROR: Error establishing connection with client. ");
					continue;
				}

				epoll_event new_event;
				new_event.events = EPOLLIN;
				new_event.data.fd = connection;
				if (epoll_ctl(epoll_instance, EPOLL_CTL_ADD, connection, &new_event) == -1) {
					mpi_error("run_server ERROR: Failed to add network polling event. ");
					close(connection); continue;
				}
			} else {
				if (events[i].events & POLL_IN) {
					/* there is incoming data from a client */
					connection_queue.add(events[i].data.fd);
					cv.notify_one();
				} if (events[i].events & (POLL_HUP | POLL_ERR)) {
					/* the client has closed this connection, so remove it */
					if (epoll_ctl(epoll_instance, EPOLL_CTL_DEL, events[i].data.fd, NULL) == -1) {
						mpi_error("run_server ERROR: Failed to remove network polling event. ");
						close(events[i].data.fd); continue;
					}
					close(events[i].data.fd);
				}
			}
		}
	}

	for (pollfd event : poll_events)
		close(event.fd);
	close(sock);
	epoll_close(epoll_instance);
	for (unsigned int i = 0; i < worker_count; i++)
		threads[i].join();
	delete[] threads;
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
	ADD_AGENT = 0,
	ADD_AGENT_RESPONSE = 1,
	MOVE = 2,
	MOVE_RESPONSE = 3,
	GET_POSITION = 4,
	GET_POSITION_RESPONSE = 5,
	STEP_RESPONSE = 6
};

struct async_server {
	std::thread server_thread;
	bool server_running;
};

inline bool receive_add_agent(socket_type& connection, simulator& sim) {
	agent_state* new_agent = sim.add_agent();
	return write(message_type::ADD_AGENT_RESPONSE, connection)
		&& write((uint64_t) new_agent, connection);
}

inline bool receive_move(socket_type& connection, simulator& sim) {
	uint64_t agent_handle;
	direction dir;
	unsigned int num_steps;
	if (!read(agent_handle, connection) && !read(dir, connection) && !read(num_steps, connection))
		return false;
	bool result = sim.move(*((agent_state*) agent_handle), dir, num_steps);
	return write(message_type::MOVE_RESPONSE, connection)
		&& write(result, connection);
}

inline bool receive_get_position(socket_type& connection, simulator& sim) {
	uint64_t agent_handle;
	if (!read(agent_handle, connection))
		return false;
	position location = sim.get_position(*((agent_state*) agent_handle));
	return write(message_type::GET_POSITION_RESPONSE, connection)
		&& write(location, connection);
}

void server_process_message(socket_type& connection, simulator& sim) {
	message_type type;
	read(connection, type);
	switch (type) {
		case message_type::ADD_AGENT:
			receive_add_agent(connection, sim); break;
		case message_type::MOVE:
			receive_move(connection, sim); break;
		case message_type::GET_POSITION:
			receive_get_position(connection, sim); break;
	}
}

bool init_server(async_server& new_server, simulator& sim,
		const char* server_address, const char* server_port,
		unsigned int connection_queue_capacity, unsigned int worker_count)
{
	std::condition_variable cv; std::mutex lock;
	auto dispatch = [&]() {
		run_server(server_address, server_port, connection_queue_capacity, worker_count, server_process_message, sim, new_server.server_running, cv);
	};
	new_server.server_running = true;
	new_server.server_thread = std::thread(dispatch);

	std::unique_lock<std::mutex> lck(lock);
	cv.wait(lck);
	if (!new_server.server_running) {
		new_server.server_thread.join();
		return false;
	}
	return true;
}

inline bool init_server(simulator& sim,
		const char* server_address, const char* server_port,
		unsigned int connection_queue_capacity, unsigned int worker_count)
{
	bool dummy = true; std::condition_variable cv;
	return run_server(server_address, server_port, connection_queue_capacity, worker_count, server_process_message, sim, dummy, cv);
}

void stop_server(async_server& server) {
	server.server_running = false;
	server.server_thread.join();
}


typedef void (*add_agent_callback)(uint64_t);
typedef void (*move_callback)(bool);
typedef void (*get_position_callback)(const position&);
typedef void (*step_callback)();

struct client_callbacks {
	add_agent_callback on_add_agent;
	move_callback on_move;
	get_position_callback on_get_position;
	step_callback on_step;
};

struct client {
	socket_type connection;
	std::thread response_listener;
	client_callbacks callbacks;
	bool client_running;
};

bool send_add_agent(client& c) {
	message_type message = message_type::ADD_AGENT;
	return send(c.connection, &message, sizeof(message), 0) != 0;
}

bool send_move(client& c, uint64_t agent_handle, direction dir, unsigned int num_steps) {
	memory_stream out = memory_stream(sizeof(message_type) + sizeof(agent_handle) + sizeof(dir) + sizeof(num_steps));
	return write(message_type::MOVE, out)
		&& write(agent_handle, out)
		&& write(dir, out)
		&& write(num_steps, out)
		&& send(c.connection, out.buffer, out.length, 0) != 0;
}

bool send_get_position(client& c, uint64_t agent_handle) {
	memory_stream out = memory_stream(sizeof(message_type) + sizeof(uint64_t));
	return write(message_type::GET_POSITION, out)
		&& write(agent_handle, out)
		&& send(c.connection, out.buffer, out.length, 0) != 0;
}

inline bool receive_add_agent_response(client& c) {
	uint64_t agent_handle;
	if (!read(agent_handle, c.connection))
		return false;
	c.callbacks.on_add_agent(agent_handle);
	return true;
}

inline bool receive_move_response(client& c) {
	bool move_success;
	if (!read(move_success, c.connection))
		return false;
	c.callbacks.on_move(move_success);
	return true;
}

inline bool receive_get_position_response(client& c) {
	position location;
	if (!read(location, c.connection))
		return false;
	c.callbacks.on_get_position(location);
	return true;
}

inline bool receive_step_response(client& c) {
	c.callbacks.on_step();
}

void run_response_listener(client& c) {
	while (c.client_running) {
		message_type type;
		read(c.connection, type);
		switch (type) {
			case message_type::ADD_AGENT_RESPONSE:
				receive_add_agent_response(c); break;
			case message_type::MOVE_RESPONSE:
				receive_move_response(c); break;
			case message_type::GET_POSITION_RESPONSE:
				receive_get_position_response(c); break;
			case message_type::STEP_RESPONSE:
				receive_step_response(c); break;
			default:
				fprintf(stderr, "run_response_listener ERROR: Received invalid message type from server.\n");
				continue;
		}
	}
}

bool init_client(client& new_client, client_callbacks callbacks,
		const char* server_address, const char* server_port)
{
	auto process_connection = [&](socket_type& connection) {
		auto dispatch = [&]() {
			run_response_listener(new_client);
		};
		new_client.connection = connection;
		new_client.response_listener = std::thread(dispatch);
	};

	new_client.client_running = true;
	new_client.callbacks = callbacks;
	return run_client(server_address, server_port, process_connection);
}

void stop_client(client& c) {
	c.client_running = false;
	c.response_listener.join();
}

#endif /* NEL_MPI_H_ */
