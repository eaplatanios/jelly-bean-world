#ifndef NEL_MPI_H_
#define NEL_MPI_H_

#include "network.h"
#include "simulator.h"

namespace nel {

using namespace core;

enum class message_type : uint64_t {
	ADD_AGENT = 0,
	ADD_AGENT_RESPONSE = 1,
	MOVE = 2,
	MOVE_RESPONSE = 3,
	GET_POSITION = 4,
	GET_POSITION_RESPONSE = 5,
	STEP_RESPONSE = 6
};

template<typename Stream>
inline bool read(message_type& type, Stream& in) {
	uint64_t v;
	if (!read(v, in)) return false;
	type = (message_type) v;
	return true;
}

template<typename Stream>
inline bool write(const message_type& type, Stream& out) {
	return write((uint64_t) type, out);
}

template<typename Stream>
inline bool print(const message_type& type, Stream& out) {
	switch (type) {
	case message_type::ADD_AGENT:    return core::print("ADD_AGENT", out);
	case message_type::MOVE:         return core::print("MOVE", out);
	case message_type::GET_POSITION: return core::print("GET_POSITION", out);

	case message_type::ADD_AGENT_RESPONSE:    return core::print("ADD_AGENT_RESPONSE", out);
	case message_type::MOVE_RESPONSE:         return core::print("MOVE_RESPONSE", out);
	case message_type::GET_POSITION_RESPONSE: return core::print("GET_POSITION_RESPONSE", out);
	case message_type::STEP_RESPONSE:         return core::print("STEP_RESPONSE", out);
	}
	fprintf(stderr, "print ERROR: Unrecognized message_type.\n");
	return false;
}

struct async_server {
	std::thread server_thread;
	socket_type server_socket;
	server_state state;
	hash_set<socket_type> client_connections;
	std::mutex connection_set_lock;

	async_server() : client_connections(1024, alloc_socket_keys) { }
};

inline bool send_message(socket_type& socket, const void* data, unsigned int length) {
	return send(socket.handle, (const char*) data, length, 0) != 0;
}

template<typename SimulatorData>
inline bool receive_add_agent(socket_type& connection, simulator<SimulatorData>& sim) {
	uint64_t new_agent = sim.add_agent();
	memory_stream mem_stream = memory_stream(sizeof(message_type) + sizeof(new_agent));
	fixed_width_stream<memory_stream> out(mem_stream);
	return write(message_type::ADD_AGENT_RESPONSE, out)
		&& write(new_agent, out)
		&& send_message(connection, mem_stream.buffer, mem_stream.length);
}

template<typename SimulatorData>
inline bool receive_move(socket_type& connection, simulator<SimulatorData>& sim) {
	uint64_t agent_id;
	direction dir;
	unsigned int num_steps;
	if (!read(agent_id, connection) || !read(dir, connection) || !read(num_steps, connection))
		return false;
	bool result = sim.move(agent_id, dir, num_steps);
	memory_stream mem_stream = memory_stream(sizeof(message_type) + sizeof(agent_id) + sizeof(result));
	fixed_width_stream<memory_stream> out(mem_stream);
	return write(message_type::MOVE_RESPONSE, out)
		&& write(agent_id, out) && write(result, out)
		&& send_message(connection, mem_stream.buffer, mem_stream.length);
}

template<typename SimulatorData>
inline bool receive_get_position(socket_type& connection, simulator<SimulatorData>& sim) {
	uint64_t agent_id;
	if (!read(agent_id, connection))
		return false;
	position location = sim.get_position(agent_id);
	memory_stream mem_stream = memory_stream(sizeof(message_type) + sizeof(agent_id) + sizeof(location));
	fixed_width_stream<memory_stream> out(mem_stream);
	return write(message_type::GET_POSITION_RESPONSE, out)
		&& write(agent_id, out) && write(location, out)
		&& send_message(connection, mem_stream.buffer, mem_stream.length);
}

template<typename SimulatorData>
void server_process_message(socket_type& connection, simulator<SimulatorData>& sim) {
	message_type type;
	if (!read(type, connection)) return;
	switch (type) {
		case message_type::ADD_AGENT:
			receive_add_agent(connection, sim); return;
		case message_type::MOVE:
			receive_move(connection, sim); return;
		case message_type::GET_POSITION:
			receive_get_position(connection, sim); return;

		case message_type::ADD_AGENT_RESPONSE:
		case message_type::MOVE_RESPONSE:
		case message_type::GET_POSITION_RESPONSE:
		case message_type::STEP_RESPONSE:
			break;
	}
	fprintf(stderr, "server_process_message WARNING: Received message with unsupported type.\n");
}

inline bool send_step_response(async_server& server) {
	std::unique_lock<std::mutex> lock(server.connection_set_lock);
	bool success = true;
	for (socket_type& client_connection : server.client_connections)
		success &= write(message_type::STEP_RESPONSE, client_connection);
	return success;
}

template<typename SimulatorData>
bool init_server(
		async_server& new_server, simulator<SimulatorData>& sim, uint16_t server_port,
		unsigned int connection_queue_capacity, unsigned int worker_count)
{
	std::condition_variable cv; std::mutex lock;
	auto dispatch = [&]() {
		run_server(new_server.server_socket, server_port,
				connection_queue_capacity, worker_count, new_server.state, cv, lock,
				new_server.client_connections, new_server.connection_set_lock,
				server_process_message<SimulatorData>, sim);
	};
	new_server.state = server_state::STARTING;
	new_server.server_thread = std::thread(dispatch);

	std::unique_lock<std::mutex> lck(lock);
	while (new_server.state == server_state::STARTING)
		cv.wait(lck);
	lck.unlock();
	if (new_server.state == server_state::STOPPING && new_server.server_thread.joinable()) {
		try {
			new_server.server_thread.join();
		} catch (...) { }
		return false;
	}
	return true;
}

template<typename SimulatorData>
inline bool init_server(simulator<SimulatorData>& sim, uint16_t server_port,
		unsigned int connection_queue_capacity, unsigned int worker_count)
{
	socket_type server_socket;
	server_state dummy = server_state::STARTING;
	std::condition_variable cv; std::mutex lock, connection_set_lock;
	hash_set<socket_type> connections(1024, alloc_socket_keys);
	return run_server(server_socket, server_port,
			connection_queue_capacity, worker_count, dummy, cv, lock,
			connections, connection_set_lock, server_process_message, sim);
}

void stop_server(async_server& server) {
	server.state = server_state::STOPPING;
	close(server.server_socket);
	if (server.server_thread.joinable()) {
		try {
			server.server_thread.join();
		} catch (...) { }
	}
}


template<typename ClientData>
struct client {
	socket_type connection;
	std::thread response_listener;
	bool client_running;
	ClientData data;
};

template<typename ClientType>
bool send_add_agent(ClientType& c) {
	message_type message = message_type::ADD_AGENT;
	return send_message(c.connection, &message, sizeof(message));
}

template<typename ClientType>
bool send_move(ClientType& c, uint64_t agent_id, direction dir, unsigned int num_steps) {
	memory_stream mem_stream = memory_stream(sizeof(message_type) + sizeof(agent_id) + sizeof(dir) + sizeof(num_steps));
	fixed_width_stream<memory_stream> out(mem_stream);
	return write(message_type::MOVE, out)
		&& write(agent_id, out)
		&& write(dir, out)
		&& write(num_steps, out)
		&& send_message(c.connection, mem_stream.buffer, mem_stream.length);
}

template<typename ClientType>
bool send_get_position(ClientType& c, uint64_t agent_id) {
	memory_stream mem_stream = memory_stream(sizeof(message_type) + sizeof(uint64_t));
	fixed_width_stream<memory_stream> out(mem_stream);
	return write(message_type::GET_POSITION, out)
		&& write(agent_id, out)
		&& send_message(c.connection, mem_stream.buffer, mem_stream.length);
}

template<typename ClientType>
inline bool receive_add_agent_response(ClientType& c) {
	uint64_t agent_id;
	fixed_width_stream<socket_type> in(c.connection);
	if (!read(agent_id, in))
		return false;
	on_add_agent(c, agent_id);
	return true;
}

template<typename ClientType>
inline bool receive_move_response(ClientType& c) {
	bool move_success;
	uint64_t agent_id;
	fixed_width_stream<socket_type> in(c.connection);
	if (!read(agent_id, in) || !read(move_success, in))
		return false;
	on_move(c, agent_id, move_success);
	return true;
}

template<typename ClientType>
inline bool receive_get_position_response(ClientType& c) {
	position location;
	uint64_t agent_id;
	fixed_width_stream<socket_type> in(c.connection);
	if (!read(agent_id, in) || !read(location, in))
		return false;
	on_get_position(c, agent_id, location);
	return true;
}

template<typename ClientType>
inline bool receive_step_response(ClientType& c) {
	on_step(c);
	return true;
}

template<typename ClientType>
void run_response_listener(ClientType& c) {
	while (c.client_running) {
		message_type type;
		bool success = read(type, c.connection);
		if (!c.client_running) {
			return; /* stop_client was called */
		} else if (!success) {
			on_lost_connection(c);
			return;
		}
		switch (type) {
			case message_type::ADD_AGENT_RESPONSE:
				receive_add_agent_response(c); continue;
			case message_type::MOVE_RESPONSE:
				receive_move_response(c); continue;
			case message_type::GET_POSITION_RESPONSE:
				receive_get_position_response(c); continue;
			case message_type::STEP_RESPONSE:
				receive_step_response(c); continue;

			case message_type::ADD_AGENT:
			case message_type::MOVE:
			case message_type::GET_POSITION:
				break;
		}
		fprintf(stderr, "run_response_listener ERROR: Received invalid message type from server %" PRId64 ".\n", (uint64_t) type);
	}
}

template<typename ClientData>
bool init_client(client<ClientData>& new_client,
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
	return run_client(server_address, server_port, process_connection);
}

template<typename ClientData>
void stop_client(client<ClientData>& c) {
	c.client_running = false;
	shutdown(c.connection.handle, 2);
	if (c.response_listener.joinable()) {
		try {
			c.response_listener.join();
		} catch (...) { }
	}
}

} /* namespace nel */

#endif /* NEL_MPI_H_ */
