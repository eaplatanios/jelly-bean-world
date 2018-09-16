#ifndef NEL_MPI_H_
#define NEL_MPI_H_

#include "network.h"
#include "simulator.h"

namespace nel {

using namespace core;

enum class message_type : uint64_t {
	ADD_AGENT = 0,
	ADD_AGENT_RESPONSE,
	MOVE,
	MOVE_RESPONSE,
	TURN,
	TURN_RESPONSE,
	GET_MAP,
	GET_MAP_RESPONSE,
	STEP_RESPONSE
};

/**
 * Reads a message_type from `in` and stores the result in `type`.
 */
template<typename Stream>
inline bool read(message_type& type, Stream& in) {
	uint64_t v;
	if (!read(v, in)) return false;
	type = (message_type) v;
	return true;
}

/**
 * Writes the given message_type `type` to the stream `out`.
 */
template<typename Stream>
inline bool write(const message_type& type, Stream& out) {
	return write((uint64_t) type, out);
}

/**
 * Prints the given message_type `type` to the stream `out`.
 */
template<typename Stream>
inline bool print(const message_type& type, Stream& out) {
	switch (type) {
	case message_type::ADD_AGENT:        return core::print("ADD_AGENT", out);
	case message_type::MOVE:             return core::print("MOVE", out);
	case message_type::TURN:             return core::print("TURN", out);
	case message_type::GET_MAP:          return core::print("GET_MAP", out);

	case message_type::ADD_AGENT_RESPONSE:        return core::print("ADD_AGENT_RESPONSE", out);
	case message_type::MOVE_RESPONSE:             return core::print("MOVE_RESPONSE", out);
	case message_type::TURN_RESPONSE:             return core::print("TURN_RESPONSE", out);
	case message_type::GET_MAP_RESPONSE:          return core::print("GET_MAP_RESPONSE", out);
	case message_type::STEP_RESPONSE:             return core::print("STEP_RESPONSE", out);
	}
	fprintf(stderr, "print ERROR: Unrecognized message_type.\n");
	return false;
}

/**
 * A structure that keeps track of additional state for each connection in
 * `async_server`. For now, it just keeps track of the agent IDs governed by
 * each connected client.
 */
struct connection_data {
	array<uint64_t> agent_ids;

	connection_data() : agent_ids(8) { }

	static inline void move(const connection_data& src, connection_data& dst) {
		core::move(src.agent_ids, dst.agent_ids);
	}

	static inline void free(connection_data& connection) {
		core::free(connection.agent_ids);
	}
};

/**
 * Initializes a new empty connection_data structure in `connection`.
 */
inline bool init(connection_data& connection) {
	return array_init(connection.agent_ids, 8);
}

/**
 * A structure containing the state of a simulator server that runs
 * asynchronously on a separate thread. The `init_server` function is
 * responsible for setting up the TCP sockets and dispatching the server
 * thread.
 */
struct async_server {
	std::thread server_thread;
	socket_type server_socket;
	server_state state;
	hash_map<socket_type, connection_data> client_connections;
	std::mutex connection_set_lock;

	async_server() : client_connections(1024, alloc_socket_keys) { }
	~async_server() { free_helper(); }

	static inline void free(async_server& server) {
		server.free_helper();
		core::free(server.client_connections);
		server.server_thread.~thread();
		server.connection_set_lock.~mutex();
	}

private:
	inline void free_helper() {
		for (auto connection : client_connections)
			core::free(connection.value);
	}
};

/**
 * Initializes the given async_server `new_server`. The `init_server` function
 * is responsible for setting up the TCP sockets and dispatching the server
 * thread.
 */
inline bool init(async_server& new_server) {
	if (!hash_map_init(new_server.client_connections, 1024, alloc_socket_keys))
		return false;
	new (&new_server.server_thread) std::thread();
	new (&new_server.connection_set_lock) std::mutex();
	return true;
}

/**
 * Writes the bytes in `data` of length `length` to the TCP socket in `socket`.
 */
inline bool send_message(socket_type& socket, const void* data, unsigned int length) {
	return send(socket.handle, (const char*) data, length, 0) != 0;
}

/* TODO: the below functions should send an error back to client upon failure */

template<typename Stream, typename SimulatorData>
inline bool receive_add_agent(
		Stream& in, socket_type& connection,
		hash_map<socket_type, connection_data>& connections,
		simulator<SimulatorData>& sim)
{
	pair<uint64_t, agent_state*> new_agent = sim.add_agent();
	if (new_agent.value != NULL)
		connections.get(connection).agent_ids.add(new_agent.key);
	memory_stream mem_stream = memory_stream(sizeof(message_type) + sizeof(new_agent.key) + sizeof(new_agent.value));
	fixed_width_stream<memory_stream> out(mem_stream);
	std::unique_lock<std::mutex> lock(new_agent.value->lock);
	return write(message_type::ADD_AGENT_RESPONSE, out)
		&& write(new_agent.key, out)
		&& (new_agent.value == NULL || write(*new_agent.value, out, sim.get_config()))
		&& send_message(connection, mem_stream.buffer, mem_stream.position);
}

template<typename Stream, typename SimulatorData>
inline bool receive_move(Stream& in, socket_type& connection, simulator<SimulatorData>& sim) {
	uint64_t agent_id = UINT64_MAX;
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
inline bool receive_turn(Stream& in, socket_type& connection, simulator<SimulatorData>& sim) {
	uint64_t agent_id = UINT64_MAX;
	direction dir;
	if (!read(agent_id, in) || !read(dir, in))
		return false;
	bool result = sim.turn(agent_id, dir);
	memory_stream mem_stream = memory_stream(sizeof(message_type) + sizeof(agent_id) + sizeof(result));
	fixed_width_stream<memory_stream> out(mem_stream);

	return write(message_type::TURN_RESPONSE, out)
		&& write(agent_id, out) && write(result, out)
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
void server_process_message(socket_type& connection,
		hash_map<socket_type, connection_data>& connections,
		simulator<SimulatorData>& sim)
{
	message_type type;
	fixed_width_stream<socket_type> in(connection);
	if (!read(type, in)) return;
	switch (type) {
		case message_type::ADD_AGENT:
			receive_add_agent(in, connection, connections, sim); return;
		case message_type::MOVE:
			receive_move(in, connection, sim); return;
		case message_type::TURN:
			receive_turn(in, connection, sim); return;
		case message_type::GET_MAP:
			receive_get_map(in, connection, sim); return;

		case message_type::ADD_AGENT_RESPONSE:
		case message_type::MOVE_RESPONSE:
		case message_type::TURN_RESPONSE:
		case message_type::GET_MAP_RESPONSE:
		case message_type::STEP_RESPONSE:
			break;
	}
	fprintf(stderr, "server_process_message WARNING: Received message with unrecognized type.\n");
}

template<typename SimulatorData>
inline bool process_new_connection(socket_type& connection,
		connection_data& data, simulator<SimulatorData>& sim)
{
	/* send the simulator time and configuration */
	memory_stream mem_stream = memory_stream(sizeof(sim.time));
	fixed_width_stream<memory_stream> out(mem_stream);
	const simulator_config& config = sim.get_config();
	if (!write(sim.time, out)
	 || !write(config, out)) {
		fprintf(stderr, "process_new_connection ERROR: Failed to send simulator time and configuration.\n");
		return false;
	}

	/* read the agent IDs owned by the new client */
	unsigned int agent_count = 0;
	fixed_width_stream<socket_type> in(connection);
	if (!read(agent_count, in)) {
		fprintf(stderr, "process_new_connection ERROR: Failed to read agent_count.\n");
		return false;
	}

	if (agent_count > 0) {
		uint64_t* agent_ids = (uint64_t*) malloc(sizeof(uint64_t) * agent_count);
		agent_state** agent_states = (agent_state**) malloc(sizeof(agent_state*) * agent_count);
		if (agent_ids == NULL || agent_states == NULL) {
			if (agent_ids != NULL) free(agent_ids);
			fprintf(stderr, "process_new_connection ERROR: Out of memory.\n");
			return false;
		} else if (!read(agent_ids, in, agent_count)) {
			fprintf(stderr, "process_new_connection ERROR: Failed to read agent_ids.\n");
			free(agent_ids); free(agent_states); return false;
		}
		data.agent_ids.append(agent_ids, agent_count);
		sim.get_agent_states(agent_states, agent_ids, agent_count);

		/* send the requested agent states to the client */
		for (unsigned int i = 0; i < agent_count; i++) {
			std::unique_lock<std::mutex> lock(agent_states[i]->lock);
			if (!write(*agent_states[i], out, config)) {
				free(agent_ids); free(agent_states);
				return false;
			}
		}
		free(agent_ids); free(agent_states);
	}
	return send_message(connection, mem_stream.buffer, mem_stream.position);
}

template<typename Stream>
constexpr bool write_extra_data(Stream& out)
{
	return true;
}

template<typename Stream, typename Data, typename... ExtraData>
inline bool write_extra_data(Stream& out,
		const Data& data, ExtraData&&... extra_data)
{
	return write(data, out) && write_extra_data(out, std::forward<ExtraData>(extra_data)...);
}

/**
 * Sends a step response to every client connected to the given `server`. This
 * function should be called whenever the simulator advances time.
 *
 * \param extra_data Any additional state to be sent to every client at the end
 * 		of the step response. For each argument of type `T`, a function
 * 		`bool write(const T&, fixed_width_stream<memory_stream>&)` should be
 * 		defined.
 */
template<typename... ExtraData>
inline bool send_step_response(
		async_server& server,
		const array<agent_state*>& agents,
		const simulator_config& config,
		ExtraData&&... extra_data)
{
	std::unique_lock<std::mutex> lock(server.connection_set_lock);
	bool success = true;
	for (const auto& client_connection : server.client_connections) {
		const array<uint64_t>& agent_ids = client_connection.value.agent_ids;
		memory_stream mem_stream = memory_stream(sizeof(message_type) + sizeof(unsigned int) +
				(unsigned int) agent_ids.length * (sizeof(uint64_t) + sizeof(agent_state)));
		fixed_width_stream<memory_stream> out(mem_stream);
		if (!write(message_type::STEP_RESPONSE, out)
		 || !write(agent_ids, out)) {
			success = false;
			continue;
		}

		bool client_success = true;
		for (uint64_t agent_id : agent_ids) {
			if (!write(*agents[(size_t) agent_id], out, config)) {
				client_success = false;
				break;
			}
		}
		if (!client_success || !write_extra_data(out, std::forward<ExtraData>(extra_data)...)) {
			success = false;
			continue;
		}
		success &= send_message(client_connection.key, mem_stream.buffer, mem_stream.position);
	}
	return success;
}

/**
 * Sets up the TCP sockets for `new_server` and dispatches the thread on which
 * the server will run.
 *
 * \param new_server The async_server structure containing the state of the new
 * 		server.
 * \param sim The simulator governed by the new server.
 * \param server_port The port to listen for new connections.
 * \param connection_queue_capacity The maximum number of simultaneous new
 * 		connections that can be handled by the server.
 * \param worker_count The number of worker threads to dispatch. They are
 * 		tasked with processing incoming message from clients.
 * \returns `true` if successful; `false` otherwise.
 */
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
				server_process_message<SimulatorData>, process_new_connection<SimulatorData>, sim);
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

/**
 * Sets up the TCP sockets for `new_server` and starts the server **on this
 * thread**. That is, the function will not return unless the server shuts
 * down.
 *
 * \param sim The simulator governed by the new server.
 * \param server_port The port to listen for new connections.
 * \param connection_queue_capacity The maximum number of simultaneous new
 * 		connections that can be handled by the server.
 * \param worker_count The number of worker threads to dispatch. They are
 * 		tasked with processing incoming message from clients.
 * \returns `true` if successful; `false` otherwise.
 */
template<typename SimulatorData>
inline bool init_server(simulator<SimulatorData>& sim, uint16_t server_port,
		unsigned int connection_queue_capacity, unsigned int worker_count)
{
	socket_type server_socket;
	server_state dummy = server_state::STARTING;
	std::condition_variable cv; std::mutex lock, connection_set_lock;
	hash_map<socket_type, connection_data> connections(1024, alloc_socket_keys);
	return run_server(server_socket, server_port, connection_queue_capacity,
			worker_count, dummy, cv, lock, connections, connection_set_lock,
			server_process_message<SimulatorData>, process_new_connection<SimulatorData>, sim);
}

/**
 * Shuts down the asynchronous server given by `server`.
 */
void stop_server(async_server& server) {
	server.state = server_state::STOPPING;
	close(server.server_socket);
	if (server.server_thread.joinable()) {
		try {
			server.server_thread.join();
		} catch (...) { }
	}
}


/**
 * A structure that contains the state of a client, which may connect to a
 * simulator server.
 *
 * \tparam ClientData A generic type that enables the storage of additional
 * 		state information in each client object.
 */
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

/**
 * Initializes a new unconnected client in `new_client`.
 */
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

/**
 * Sends an `add_agent` message to the server from the client `c`. Once the
 * server responds, the function
 * `on_add_agent(ClientType&, uint64_t, agent_state&)` will be invoked, where
 * the first argument is `c`, the second is the ID of the new agent
 * (which will be UINT64_MAX upon error), and the third is the state of the new
 * agent.
 *
 * \returns `true` if the sending is successful; `false` otherwise.
 */
template<typename ClientType>
bool send_add_agent(ClientType& c) {
	message_type message = message_type::ADD_AGENT;
	return send_message(c.connection, &message, sizeof(message));
}

/**
 * Sends a `move` message to the server from the client `c`. Once the server
 * responds, the function `on_move(ClientType&, uint64_t, bool)` will be
 * invoked, where the first argument is `c`, the second is `agent_id`, and the
 * third is whether the move was successfully enqueued by the server.
 *
 * \returns `true` if the sending is successful; `false` otherwise.
 */
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

/**
 * Sends a `turn` message to the server from the client `c`. Once the server
 * responds, the function `on_turn(ClientType&, uint64_t, bool)` will be
 * invoked, where the first argument is `c`, the second is `agent_id`, and the
 * third is whether the turn was successfully enqueued by the server.
 *
 * \returns `true` if the sending is successful; `false` otherwise.
 */
template<typename ClientType>
bool send_turn(ClientType& c, uint64_t agent_id, direction dir) {
	memory_stream mem_stream = memory_stream(sizeof(message_type) + sizeof(agent_id) + sizeof(dir));
	fixed_width_stream<memory_stream> out(mem_stream);
	return write(message_type::TURN, out)
		&& write(agent_id, out)
		&& write(dir, out)
		&& send_message(c.connection, mem_stream.buffer, mem_stream.position);
}

/**
 * Sends an `get_map` message to the server from the client `c`. Once the
 * server responds, the function
 * `on_get_map(ClientType&, hash_map<position, patch_state>*)` will be invoked,
 * where the first argument is `c`, and the second is a pointer to a map
 * containing the state information of the retrieved patches. Memory ownership
 * of the hash_map is passed to `on_get_map`.
 *
 * \param bottom_left The bottom-left corner of the bounding box containing the
 * 		patches we wish to retrieve.
 * \param top_right The top-right corner of the bounding box containing the
 * 		patches we wish to retrieve.
 * \returns `true` if the sending is successful; `false` otherwise.
 */
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
	agent_state& state = *((agent_state*) alloca(sizeof(agent_state)));
	fixed_width_stream<socket_type> in(c.connection);
	if (!read(agent_id, in)) return false;
	if (agent_id != UINT64_MAX && !read(state, in, c.config))
		return false;
	on_add_agent(c, agent_id, state);
	if (agent_id != UINT64_MAX) free(state);
	return true;
}

template<typename ClientType>
inline bool receive_move_response(ClientType& c) {
	bool move_success;
	uint64_t agent_id = 0;
	fixed_width_stream<socket_type> in(c.connection);
	if (!read(agent_id, in) || !read(move_success, in))
		return false;
	on_move(c, agent_id, move_success);
	return true;
}

template<typename ClientType>
inline bool receive_turn_response(ClientType& c) {
	bool turn_success;
	uint64_t agent_id = 0;
	fixed_width_stream<socket_type> in(c.connection);
	if (!read(agent_id, in) || !read(turn_success, in))
		return false;
	on_turn(c, agent_id, turn_success);
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
	/* ownership of `patches` is passed to the callee */
	on_get_map(c, patches);
	return true;
}

template<typename ClientType>
inline bool receive_step_response(ClientType& c) {
	array<uint64_t>& agent_ids = *((array<uint64_t>*) alloca(sizeof(array<uint64_t>)));

	fixed_width_stream<socket_type> in(c.connection);
	if (!read(agent_ids, in)) return false;

	agent_state* agents = (agent_state*) malloc(sizeof(agent_state) * agent_ids.length);
	if (agents == NULL) {
		fprintf(stderr, "receive_step_response ERROR: Out of memory.\n");
		free(agent_ids); return false;
	}

	for (unsigned int i = 0; i < agent_ids.length; i++) {
		if (!read(agents[i], in, c.config)) {
			for (unsigned int j = 0; j < i; j++) free(agents[j]);
			free(agents); free(agent_ids); return false;
		}
	}

	on_step(c, (const array<uint64_t>&) agent_ids, (const agent_state*) agents);
	for (unsigned int i = 0; i < agent_ids.length; i++)
		free(agents[i]);
	free(agent_ids);
	free(agents);
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
			case message_type::TURN_RESPONSE:
				receive_turn_response(c); continue;
			case message_type::GET_MAP_RESPONSE:
				receive_get_map_response(c); continue;
			case message_type::STEP_RESPONSE:
				receive_step_response(c); continue;

			case message_type::ADD_AGENT:
			case message_type::MOVE:
			case message_type::TURN:
			case message_type::GET_MAP:
				break;
		}
		fprintf(stderr, "run_response_listener ERROR: Received invalid message type from server %" PRId64 ".\n", (uint64_t) type);
	}
}

/**
 * Attempts to connect the given client `new_client` to the server at
 * `server_address:server_port`. A separate thread will be dispatched to listen
 * for responses from the server. `stop_client` should be used to disconnect
 * from the server and stop the listener thread.
 *
 * \param new_client The client with which to attempt the connection.
 * \param server_address A null-terminated string containing the server
 * 		address.
 * \param server_port A null-terminated string containing the server port.
 * \param agent_ids An array of agent IDs governed by this client, of length
 * 		`agent_count`.
 * \param agent_states An array of length `agent_count` to which this function
 * 		will write the states of the agents whose IDs are given by the parallel
 * 		array `agent_ids`.
 * \param agent_count The lengths of `agent_ids` and `agent_states`.
 * \returns The simulator time if successful; `UINT64_MAX` otherwise.
 */
template<typename ClientData>
uint64_t init_client(client<ClientData>& new_client,
		const char* server_address, const char* server_port,
		const uint64_t* agent_ids, agent_state* agent_states,
		unsigned int agent_count)
{
	uint64_t simulator_time;
	auto process_connection = [&](socket_type& connection)
	{
		new_client.connection = connection;

		/* send the list of agent IDs owned by this client */
		memory_stream mem_stream = memory_stream(sizeof(agent_count) + sizeof(uint64_t) * agent_count);
		fixed_width_stream<memory_stream> out(mem_stream);
		if (!write(agent_count, out)
		 || !write(agent_ids, out, agent_count)
		 || !send_message(connection, mem_stream.buffer, mem_stream.position))
		{
			fprintf(stderr, "init_client ERROR: Error requesting agent states.\n");
			stop_client(new_client);
			return false;
		}

		/* read the simulator time and configuration */
		fixed_width_stream<socket_type> in(connection);
		simulator_config& config = *((simulator_config*) alloca(sizeof(simulator_config)));
		if (!read(simulator_time, in)
		 || !read(config, in))
		{
			fprintf(stderr, "init_client ERROR: Error receiving simulator time and configuration.\n");
			stop_client(new_client);
			return false;
		}
		swap(new_client.config, config);
		free(config); /* free the old configuration */

		/* read the agent states for the requested agent IDs */
		for (unsigned int i = 0; i < agent_count; i++) {
			if (!read(agent_states[i], in, new_client.config)) {
				for (unsigned int j = 0; j < i; j++) free(agent_states[j]);
				return false;
			}
		}

		auto dispatch = [&]() {
			run_response_listener(new_client);
		};
		new_client.response_listener = std::thread(dispatch);
		return true;
	};

	new_client.client_running = true;
	if (!run_client(server_address, server_port, process_connection))
		return UINT64_MAX;
	else return simulator_time;
}

/**
 * Attempts to connect the given client `new_client` to the server at
 * `server_address:server_port`. A separate thread will be dispatched to listen
 * for responses from the server. `stop_client` should be used to disconnect
 * from the server and stop the listener thread.
 *
 * \param new_client The client with which to attempt the connection.
 * \param server_address A null-terminated string containing the server
 * 		address.
 * \param server_port The server port.
 * \param agent_ids An array of agent IDs governed by this client, of length
 * 		`agent_count`.
 * \param agent_states An array of length `agent_count` to which this function
 * 		will write the states of the agents whose IDs are given by the parallel
 * 		array `agent_ids`.
 * \param agent_count The lengths of `agent_ids` and `agent_states`.
 * \returns The simulator time if successful; `UINT64_MAX` otherwise.
 */
template<typename ClientData>
inline uint64_t init_client(client<ClientData>& new_client,
		const char* server_address, uint16_t server_port,
		const uint64_t* agent_ids, agent_state* agent_states,
		unsigned int agent_count)
{
	constexpr static unsigned int BUFFER_SIZE = 8;
	char port_str[BUFFER_SIZE];
	if (snprintf(port_str, BUFFER_SIZE, "%u", server_port) > (int) BUFFER_SIZE - 1)
		return false;

	return init_client(new_client, server_address, port_str, agent_ids, agent_states, agent_count);
}

/**
 * Disconnects the given client `c` from the server.
 */
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
