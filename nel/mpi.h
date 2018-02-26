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
	GET_CONFIG = 6,
	GET_CONFIG_RESPONSE = 7,
	GET_SCENT = 8,
	GET_SCENT_RESPONSE = 9,
	GET_VISION = 10,
	GET_VISION_RESPONSE = 11,
	GET_COLLECTED_ITEMS = 12,
	GET_COLLECTED_ITEMS_RESPONSE = 13,
	GET_MAP = 14,
	GET_MAP_RESPONSE = 15,
	STEP_RESPONSE = 16
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
	case message_type::ADD_AGENT:           return core::print("ADD_AGENT", out);
	case message_type::MOVE:                return core::print("MOVE", out);
	case message_type::GET_POSITION:        return core::print("GET_POSITION", out);
	case message_type::GET_CONFIG:          return core::print("GET_CONFIG", out);
	case message_type::GET_SCENT:           return core::print("GET_SCENT", out);
	case message_type::GET_VISION:          return core::print("GET_VISION", out);
	case message_type::GET_COLLECTED_ITEMS: return core::print("GET_COLLECTED_ITEMS", out);
	case message_type::GET_MAP:             return core::print("GET_MAP", out);

	case message_type::ADD_AGENT_RESPONSE:           return core::print("ADD_AGENT_RESPONSE", out);
	case message_type::MOVE_RESPONSE:                return core::print("MOVE_RESPONSE", out);
	case message_type::GET_POSITION_RESPONSE:        return core::print("GET_POSITION_RESPONSE", out);
	case message_type::GET_CONFIG_RESPONSE:          return core::print("GET_CONFIG_RESPONSE", out);
	case message_type::GET_SCENT_RESPONSE:           return core::print("GET_SCENT_RESPONSE", out);
	case message_type::GET_VISION_RESPONSE:          return core::print("GET_VISION_RESPONSE", out);
	case message_type::GET_COLLECTED_ITEMS_RESPONSE: return core::print("GET_VISION_RESPONSE", out);
	case message_type::GET_MAP_RESPONSE:             return core::print("GET_MAP_RESPONSE", out);
	case message_type::STEP_RESPONSE:                return core::print("STEP_RESPONSE", out);
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

	static inline void free(async_server& server) {
		core::free(server.client_connections);
		server.server_thread.~thread();
		server.connection_set_lock.~mutex();
	}
};

inline bool init(async_server& new_server) {
	if (!hash_set_init(new_server.client_connections, 1024))
		return false;
	new (&new_server.server_thread) std::thread();
	new (&new_server.connection_set_lock) std::mutex();
	return true;
}

inline bool send_message(socket_type& socket, const void* data, unsigned int length) {
	return send(socket.handle, (const char*) data, length, 0) != 0;
}

template<typename Stream, typename SimulatorData>
inline bool receive_add_agent(Stream& in, socket_type& connection, simulator<SimulatorData>& sim) {
	uint64_t new_agent = sim.add_agent();
	memory_stream mem_stream = memory_stream(sizeof(message_type) + sizeof(new_agent));
	fixed_width_stream<memory_stream> out(mem_stream);
	return write(message_type::ADD_AGENT_RESPONSE, out)
		&& write(new_agent, out)
		&& send_message(connection, mem_stream.buffer, mem_stream.position);
}

template<typename Stream, typename SimulatorData>
inline bool receive_move(Stream& in, socket_type& connection, simulator<SimulatorData>& sim) {
	uint64_t agent_id;
	direction dir;
	unsigned int num_steps;
	if (!read(agent_id, in) || !read(dir, in) || !read(num_steps, in))
		return false;
	bool result = sim.move(agent_id, dir, num_steps);
	memory_stream mem_stream = memory_stream(sizeof(message_type) + sizeof(agent_id) + sizeof(result));
	fixed_width_stream<memory_stream> out(mem_stream);
	return write(message_type::MOVE_RESPONSE, out)
		&& write(agent_id, out) && write(result, out)
		&& send_message(connection, mem_stream.buffer, mem_stream.position);
}

template<typename Stream, typename SimulatorData>
inline bool receive_get_position(Stream& in, socket_type& connection, simulator<SimulatorData>& sim) {
	uint64_t agent_id;
	if (!read(agent_id, in))
		return false;
	position location = sim.get_position(agent_id);
	memory_stream mem_stream = memory_stream(sizeof(message_type) + sizeof(agent_id) + sizeof(location));
	fixed_width_stream<memory_stream> out(mem_stream);
	return write(message_type::GET_POSITION_RESPONSE, out)
		&& write(agent_id, out) && write(location, out)
		&& send_message(connection, mem_stream.buffer, mem_stream.position);
}

template<typename Stream, typename SimulatorData>
inline bool receive_get_config(Stream& in, socket_type& connection, simulator<SimulatorData>& sim) {
	const simulator_config& config = sim.get_config();
	memory_stream mem_stream = memory_stream(sizeof(message_type) + sizeof(config));
	fixed_width_stream<memory_stream> out(mem_stream);
	return write(message_type::GET_CONFIG_RESPONSE, out)
		&& write(config, out)
		&& send_message(connection, mem_stream.buffer, mem_stream.position);
}

template<typename Stream, typename SimulatorData>
inline bool receive_get_scent(Stream& in, socket_type& connection, simulator<SimulatorData>& sim) {
	uint64_t agent_id;
	if (!read(agent_id, in))
		return false;
	const float* scent = sim.get_scent(agent_id);
	memory_stream mem_stream = memory_stream(sizeof(message_type)
			+ sizeof(agent_id) + (sizeof(float) * sim.get_config().scent_dimension));
	fixed_width_stream<memory_stream> out(mem_stream);
	return write(message_type::GET_SCENT_RESPONSE, out)
		&& write(agent_id, out) && write(scent, out, sim.get_config().scent_dimension)
		&& send_message(connection, mem_stream.buffer, mem_stream.position);
}

template<typename Stream, typename SimulatorData>
inline bool receive_get_vision(Stream& in, socket_type& connection, simulator<SimulatorData>& sim) {
	uint64_t agent_id;
	if (!read(agent_id, in))
		return false;
	const float* vision = sim.get_vision(agent_id);
	unsigned int vision_size = (2 * sim.get_config().vision_range + 1)
			* (2 * sim.get_config().vision_range + 1) * sim.get_config().color_dimension;
	memory_stream mem_stream = memory_stream(sizeof(message_type)
			+ sizeof(agent_id) + (sizeof(float) * vision_size));
	fixed_width_stream<memory_stream> out(mem_stream);
	return write(message_type::GET_VISION_RESPONSE, out)
		&& write(agent_id, out) && write(vision, out, vision_size)
		&& send_message(connection, mem_stream.buffer, mem_stream.position);
}

template<typename Stream, typename SimulatorData>
inline bool receive_get_collected_items(Stream& in, socket_type& connection, simulator<SimulatorData>& sim) {
	uint64_t agent_id;
	if (!read(agent_id, in))
		return false;
	const unsigned int* items = sim.get_collected_items(agent_id);
	memory_stream mem_stream = memory_stream((unsigned int) (sizeof(message_type)
			+ sizeof(agent_id) + (sizeof(unsigned int) * sim.get_config().item_types.length)));
	fixed_width_stream<memory_stream> out(mem_stream);
	return write(message_type::GET_COLLECTED_ITEMS_RESPONSE, out)
		&& write(agent_id, out) && write(items, out, (unsigned int) sim.get_config().item_types.length)
		&& send_message(connection, mem_stream.buffer, mem_stream.position);
}

template<typename Stream, typename SimulatorData>
inline bool receive_get_map(Stream& in, socket_type& connection, simulator<SimulatorData>& sim) {
	position bottom_left, top_right;
	if (!read(bottom_left, in) || !read(top_right, in))
		return false;

	hash_map<position, patch_state> patches(32);
	if (!sim.get_map(bottom_left, top_right, patches)) {
		for (auto entry : patches)
			free(entry.value);
		patches.clear();
	}

	default_scribe scribe;
	memory_stream mem_stream = memory_stream(sizeof(message_type) + sizeof(hash_map<position, patch_state>));
	fixed_width_stream<memory_stream> out(mem_stream);
	if (!write(message_type::GET_MAP_RESPONSE, out)
	 || !write(patches, out, scribe, sim.get_config())
	 || !send_message(connection, mem_stream.buffer, mem_stream.position))
		return false;
	for (auto entry : patches)
		free(entry.value);
	return true;
}

template<typename SimulatorData>
void server_process_message(socket_type& connection, simulator<SimulatorData>& sim) {
	message_type type;
	fixed_width_stream<socket_type> in(connection);
	if (!read(type, in)) return;
	switch (type) {
		case message_type::ADD_AGENT:
			receive_add_agent(in, connection, sim); return;
		case message_type::MOVE:
			receive_move(in, connection, sim); return;
		case message_type::GET_POSITION:
			receive_get_position(in, connection, sim); return;
		case message_type::GET_CONFIG:
			receive_get_config(in, connection, sim); return;
		case message_type::GET_SCENT:
			receive_get_scent(in, connection, sim); return;
		case message_type::GET_VISION:
			receive_get_scent(in, connection, sim); return;
		case message_type::GET_COLLECTED_ITEMS:
			receive_get_collected_items(in, connection, sim); return;
		case message_type::GET_MAP:
			receive_get_map(in, connection, sim); return;

		case message_type::ADD_AGENT_RESPONSE:
		case message_type::MOVE_RESPONSE:
		case message_type::GET_POSITION_RESPONSE:
		case message_type::GET_CONFIG_RESPONSE:
		case message_type::GET_SCENT_RESPONSE:
		case message_type::GET_VISION_RESPONSE:
		case message_type::GET_COLLECTED_ITEMS_RESPONSE:
		case message_type::GET_MAP_RESPONSE:
		case message_type::STEP_RESPONSE:
			break;
	}
	fprintf(stderr, "server_process_message WARNING: Received message with unrecognized type.\n");
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
	simulator_config config;
	ClientData data;

	static inline void free(client<ClientData>& c) {
		c.response_listener.~thread();
		core::free(c.config);
		core::free(c.data);
	}
};

template<typename ClientData>
inline bool init(client<ClientData>& new_client) {
	if (!init(new_client.config)) {
		return false;
	} else if (!init(new_client.data)) {
		free(new_client.config);
		return false;
	}
	new (&new_client.response_listener) std::thread();
	return true;
}

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
		&& send_message(c.connection, mem_stream.buffer, mem_stream.position);
}

template<typename ClientType>
bool send_get_position(ClientType& c, uint64_t agent_id) {
	memory_stream mem_stream = memory_stream(sizeof(message_type) + sizeof(uint64_t));
	fixed_width_stream<memory_stream> out(mem_stream);
	return write(message_type::GET_POSITION, out)
		&& write(agent_id, out)
		&& send_message(c.connection, mem_stream.buffer, mem_stream.position);
}

template<typename ClientType>
bool send_get_config(ClientType& c) {
	message_type message = message_type::GET_CONFIG;
	return send_message(c.connection, &message, sizeof(message));
}

template<typename ClientType>
bool send_get_scent(ClientType& c, uint64_t agent_id) {
	memory_stream mem_stream = memory_stream(sizeof(message_type) + sizeof(uint64_t));
	fixed_width_stream<memory_stream> out(mem_stream);
	return write(message_type::GET_SCENT, out)
		&& write(agent_id, out)
		&& send_message(c.connection, mem_stream.buffer, mem_stream.position);
}

template<typename ClientType>
bool send_get_vision(ClientType& c, uint64_t agent_id) {
	memory_stream mem_stream = memory_stream(sizeof(message_type) + sizeof(uint64_t));
	fixed_width_stream<memory_stream> out(mem_stream);
	return write(message_type::GET_VISION, out)
		&& write(agent_id, out)
		&& send_message(c.connection, mem_stream.buffer, mem_stream.position);
}

template<typename ClientType>
bool send_get_collected_items(ClientType& c, uint64_t agent_id) {
	memory_stream mem_stream = memory_stream(sizeof(message_type) + sizeof(uint64_t));
	fixed_width_stream<memory_stream> out(mem_stream);
	return write(message_type::GET_COLLECTED_ITEMS, out)
		&& write(agent_id, out)
		&& send_message(c.connection, mem_stream.buffer, mem_stream.position);
}

template<typename ClientType>
bool send_get_map(ClientType& c, position bottom_left, position top_right) {
	memory_stream mem_stream = memory_stream(sizeof(message_type) + 2 * sizeof(position));
	fixed_width_stream<memory_stream> out(mem_stream);
	return write(message_type::GET_MAP, out)
		&& write(bottom_left, out) && write(top_right, out)
		&& send_message(c.connection, mem_stream.buffer, mem_stream.position);
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
inline bool receive_get_config_response(ClientType& c) {
	simulator_config& config = *((simulator_config*) alloca(sizeof(simulator_config)));
	fixed_width_stream<socket_type> in(c.connection);
	if (!read(config, in)) {
		fprintf(stderr, "receive_get_config_response ERROR: Unable to deserialize configuration.\n");
		return false;
	}
	swap(c.config, config);
	free(config); /* free the old configuration */
	return true;
}

template<typename ClientType>
inline bool receive_get_scent_response(ClientType& c) {
	uint64_t agent_id;
	fixed_width_stream<socket_type> in(c.connection);
	float* scent = (float*) malloc(sizeof(float) * c.config.scent_dimension);
	if (scent == NULL) {
		fprintf(stderr, "receive_get_scent_response ERROR: Out of memory.\n");
		return false;
	} else if (!read(agent_id, in) || !read(scent, in, c.config.scent_dimension)) {
		free(scent); return false;
	}
	on_get_scent(c, agent_id, scent);
	return true;
}

template<typename ClientType>
inline bool receive_get_vision_response(ClientType& c) {
	uint64_t agent_id;
	fixed_width_stream<socket_type> in(c.connection);
	unsigned int vision_size = (2 * c.config.vision_range + 1) * (2 * c.config.vision_range + 1) * c.config.color_dimension;
	float* vision = (float*) malloc(sizeof(float) * vision_size);
	if (vision == NULL) {
		fprintf(stderr, "receive_get_vision_response ERROR: Out of memory.\n");
		return false;
	} else if (!read(agent_id, in) || !read(vision, in, vision_size)) {
		free(vision); return false;
	}
	on_get_vision(c, agent_id, vision);
	return true;
}

template<typename ClientType>
inline bool receive_get_collected_items_response(ClientType& c) {
	uint64_t agent_id;
	fixed_width_stream<socket_type> in(c.connection);
	unsigned int* items = (unsigned int*) malloc(sizeof(unsigned int) * c.config.item_types.length);
	if (items == NULL) {
		fprintf(stderr, "receive_get_collected_items_response ERROR: Out of memory.\n");
		return false;
	} else if (!read(agent_id, in) || !read(items, in, (unsigned int) c.config.item_types.length)) {
		free(items); return false;
	}
	on_get_collected_items(c, agent_id, items);
	return true;
}

template<typename ClientType>
inline bool receive_get_map_response(ClientType& c) {
	default_scribe scribe;
	fixed_width_stream<socket_type> in(c.connection);
	hash_map<position, patch_state>* patches =
			(hash_map<position, patch_state>*) malloc(sizeof(hash_map<position, patch_state>));
	if (patches == NULL) {
		fprintf(stderr, "receive_get_map_response ERROR: Out of memory.\n");
		return false;
	} else if (!read(*patches, in, alloc_position_keys, scribe, c.config)) {
		free(patches); return false;
	}
	on_get_map(c, patches);
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
			case message_type::GET_CONFIG_RESPONSE:
				receive_get_config_response(c); continue;
			case message_type::GET_SCENT_RESPONSE:
				receive_get_scent_response(c); continue;
			case message_type::GET_VISION_RESPONSE:
				receive_get_vision_response(c); continue;
			case message_type::GET_COLLECTED_ITEMS_RESPONSE:
				receive_get_collected_items_response(c); continue;
			case message_type::GET_MAP_RESPONSE:
				receive_get_map_response(c); continue;
			case message_type::STEP_RESPONSE:
				receive_step_response(c); continue;

			case message_type::ADD_AGENT:
			case message_type::MOVE:
			case message_type::GET_POSITION:
			case message_type::GET_CONFIG:
			case message_type::GET_SCENT:
			case message_type::GET_VISION:
			case message_type::GET_COLLECTED_ITEMS:
			case message_type::GET_MAP:
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
	if (!run_client(server_address, server_port, process_connection))
		return false;
	if (!send_get_config(new_client)) {
		stop_client(new_client);
		return false;
	}
	return true;
}

template<typename ClientData>
inline bool init_client(client<ClientData>& new_client,
		const char* server_address, uint16_t server_port)
{
	constexpr static unsigned int BUFFER_SIZE = 8;
	char port_str[BUFFER_SIZE];
	if (snprintf(port_str, BUFFER_SIZE, "%u", server_port) > (int) BUFFER_SIZE - 1)
		return false;

	return init_client(new_client, server_address, port_str);
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
