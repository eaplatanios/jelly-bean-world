/**
 * Copyright 2019, The Jelly Bean World Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not
 * use this file except in compliance with the License. You may obtain a copy of
 * the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

#ifndef JBW_MPI_H_
#define JBW_MPI_H_

#include "network.h"
#include "simulator.h"

namespace jbw {

using namespace core;

constexpr uint64_t NEW_CLIENT_REQUEST = 0;

enum class message_type : uint64_t {
	ADD_AGENT = 0,
	ADD_AGENT_RESPONSE,
	MOVE,
	MOVE_RESPONSE,
	TURN,
	TURN_RESPONSE,
	DO_NOTHING,
	DO_NOTHING_RESPONSE,
	GET_MAP,
	GET_MAP_RESPONSE,
	SET_ACTIVE,
	SET_ACTIVE_RESPONSE,
	IS_ACTIVE,
	IS_ACTIVE_RESPONSE,
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
	case message_type::DO_NOTHING:       return core::print("DO_NOTHING", out);
	case message_type::GET_MAP:          return core::print("GET_MAP", out);
	case message_type::SET_ACTIVE:       return core::print("SET_ACTIVE", out);
	case message_type::IS_ACTIVE:        return core::print("IS_ACTIVE", out);

	case message_type::ADD_AGENT_RESPONSE:        return core::print("ADD_AGENT_RESPONSE", out);
	case message_type::MOVE_RESPONSE:             return core::print("MOVE_RESPONSE", out);
	case message_type::TURN_RESPONSE:             return core::print("TURN_RESPONSE", out);
	case message_type::DO_NOTHING_RESPONSE:       return core::print("DO_NOTHING_RESPONSE", out);
	case message_type::GET_MAP_RESPONSE:          return core::print("GET_MAP_RESPONSE", out);
	case message_type::SET_ACTIVE_RESPONSE:       return core::print("SET_ACTIVE_RESPONSE", out);
	case message_type::IS_ACTIVE_RESPONSE:        return core::print("IS_ACTIVE_RESPONSE", out);
	case message_type::STEP_RESPONSE:             return core::print("STEP_RESPONSE", out);
	}
	fprintf(stderr, "print ERROR: Unrecognized message_type.\n");
	return false;
}

struct client_info {
	uint64_t id;

	static inline void move(const client_info& src, client_info& dst) { dst.id = src.id; }
	static inline void free(client_info& info) { }
};

inline bool init(client_info& info) {
	info.id = 0;
	return true;
}

/**
 * A structure that keeps track of additional state for the MPI server.
 */
struct server_state {
	hash_map<uint64_t, array<uint64_t>> agent_ids;

	server_state() : agent_ids(16) { }
	~server_state() { free_helper(); }

	static inline void swap(server_state& first, server_state& second) {
		core::swap(first.agent_ids, second.agent_ids);
	}

	static inline void free(server_state& state) {
		state.free_helper();
		core::free(state.agent_ids);
	}

private:
	inline void free_helper() {
		for (auto entry : agent_ids)
			core::free(entry.value);
	}
};

bool init(server_state& state) {
	return hash_map_init(state.agent_ids, 16);
}

template<typename Stream>
bool read(server_state& state, Stream& in) {
	return read(state.agent_ids, in);
}

template<typename Stream>
bool write(const server_state& state, Stream& out) {
	return write(state.agent_ids, out);
}

/**
 * A structure containing the state of a simulator server that runs
 * synchronously on the current thread. The `init_server` function is
 * responsible for setting up the TCP sockets and dispatching the server
 * thread.
 */
struct sync_server {
	server_state state;
	hash_map<socket_type, client_info> client_connections;
	std::mutex connection_set_lock;

	sync_server() : client_connections(1024, alloc_socket_keys) { }
	~sync_server() { free_helper(); }

	static inline void free(sync_server& server) {
		server.free_helper();
		core::free(server.state);
	}

private:
	inline void free_helper() {
		for (auto connection : client_connections)
			core::free(connection.value);
	}
};

/**
 * Initializes the given sync_server `new_server`. The `init_server` function
 * is responsible for setting up the TCP sockets and dispatching the server
 * thread.
 */
inline bool init(sync_server& new_server) {
	if (!init(new_server.state)) {
		return false;
	} else if (!hash_map_init(new_server.client_connections, 1024, alloc_socket_keys)) {
		free(new_server.state); return false;
	}
	return true;
}

/**
 * A structure containing the state of a simulator server that runs
 * asynchronously on a separate thread. The `init_server` function is
 * responsible for setting up the TCP sockets and dispatching the server
 * thread.
 */
struct async_server {
	server_state state;
	std::thread server_thread;
	socket_type server_socket;
	server_status status;
	hash_map<socket_type, client_info> client_connections;
	std::mutex connection_set_lock;

	async_server() : client_connections(1024, alloc_socket_keys) { }
	~async_server() { free_helper(); }

	static inline void free(async_server& server) {
		server.free_helper();
		core::free(server.client_connections);
		core::free(server.state);
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
	if (!init(new_server.state)) {
		return false;
	} else if (!hash_map_init(new_server.client_connections, 1024, alloc_socket_keys)) {
		free(new_server.state); return false;
	}
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

enum class mpi_response : uint8_t {
	FAILURE = 0,
	SUCCESS,
	INVALID_AGENT_ID,
	SERVER_PARSE_MESSAGE_ERROR,
	CLIENT_PARSE_MESSAGE_ERROR
};

/**
 * Reads an mpi_response from `in` and stores the result in `response`.
 */
template<typename Stream>
inline bool read(mpi_response& response, Stream& in) {
	uint8_t v;
	if (!read(v, in)) return false;
	response = (mpi_response) v;
	return true;
}

/**
 * Writes the given mpi_response `response` to the stream `out`.
 */
template<typename Stream>
inline bool write(const mpi_response& response, Stream& out) {
	return write((uint8_t) response, out);
}

template<typename Stream, typename SimulatorData>
inline bool receive_add_agent(
		Stream& in, socket_type& connection,
		array<uint64_t>& agent_ids,
		simulator<SimulatorData>& sim)
{
	pair<uint64_t, agent_state*> new_agent = sim.add_agent();
	if (new_agent.value != NULL)
		agent_ids.add(new_agent.key);
	memory_stream mem_stream = memory_stream(sizeof(message_type) + sizeof(new_agent.key) + sizeof(new_agent.value));
	fixed_width_stream<memory_stream> out(mem_stream);
	std::unique_lock<std::mutex> lock(new_agent.value->lock);
	return write(message_type::ADD_AGENT_RESPONSE, out)
		&& write(new_agent.key, out)
		&& (new_agent.value == NULL || write(*new_agent.value, out, sim.get_config()))
		&& send_message(connection, mem_stream.buffer, mem_stream.position);
}

template<typename Stream, typename SimulatorData>
inline bool receive_move(
		Stream& in, socket_type& connection,
		const array<uint64_t>& agent_ids,
		simulator<SimulatorData>& sim)
{
	uint64_t agent_id = UINT64_MAX;
	direction dir;
	unsigned int num_steps;
	mpi_response response;
	bool success = true;
	if (!read(agent_id, in) || !read(dir, in) || !read(num_steps, in)) {
		response = mpi_response::SERVER_PARSE_MESSAGE_ERROR;
		success = false;
	} else {
		if (!agent_ids.contains(agent_id))
			response = mpi_response::INVALID_AGENT_ID;
		else if (sim.move(agent_id, dir, num_steps))
			response = mpi_response::SUCCESS;
		else response = mpi_response::FAILURE;
	}
	memory_stream mem_stream = memory_stream(sizeof(message_type) + sizeof(agent_id) + sizeof(response));
	fixed_width_stream<memory_stream> out(mem_stream);

	success &= write(message_type::MOVE_RESPONSE, out)
			&& write(agent_id, out) && write(response, out)
			&& send_message(connection, mem_stream.buffer, mem_stream.position);
	return success;
}

template<typename Stream, typename SimulatorData>
inline bool receive_turn(
		Stream& in, socket_type& connection,
		const array<uint64_t>& agent_ids,
		simulator<SimulatorData>& sim)
{
	uint64_t agent_id = UINT64_MAX;
	direction dir;
	mpi_response response;
	bool success = true;
	if (!read(agent_id, in) || !read(dir, in)) {
		response = mpi_response::SERVER_PARSE_MESSAGE_ERROR;
		success = false;
	} else {
		if (!agent_ids.contains(agent_id))
			response = mpi_response::INVALID_AGENT_ID;
		else if (sim.turn(agent_id, dir))
			response = mpi_response::SUCCESS;
		else response = mpi_response::FAILURE;
	}
	memory_stream mem_stream = memory_stream(sizeof(message_type) + sizeof(agent_id) + sizeof(response));
	fixed_width_stream<memory_stream> out(mem_stream);

	success &= write(message_type::TURN_RESPONSE, out)
			&& write(agent_id, out) && write(response, out)
			&& send_message(connection, mem_stream.buffer, mem_stream.position);
	return success;
}

template<typename Stream, typename SimulatorData>
inline bool receive_do_nothing(
		Stream& in, socket_type& connection,
		const array<uint64_t>& agent_ids,
		simulator<SimulatorData>& sim)
{
	uint64_t agent_id = UINT64_MAX;
	mpi_response response;
	bool success = true;
	if (!read(agent_id, in)) {
		response = mpi_response::SERVER_PARSE_MESSAGE_ERROR;
		success = false;
	} else {
		if (!agent_ids.contains(agent_id))
			response = mpi_response::INVALID_AGENT_ID;
		else if (sim.do_nothing(agent_id))
			response = mpi_response::SUCCESS;
		else response = mpi_response::FAILURE;
	}
	memory_stream mem_stream = memory_stream(sizeof(message_type) + sizeof(agent_id) + sizeof(response));
	fixed_width_stream<memory_stream> out(mem_stream);

	success &= write(message_type::DO_NOTHING_RESPONSE, out)
			&& write(agent_id, out) && write(response, out)
			&& send_message(connection, mem_stream.buffer, mem_stream.position);
	return success;
}

template<typename Stream, typename SimulatorData>
inline bool receive_get_map(
		Stream& in, socket_type& connection,
		simulator<SimulatorData>& sim)
{
	position bottom_left, top_right;
	mpi_response response;
	hash_map<position, patch_state> patches(32, alloc_position_keys);
	bool success = true;
	if (!read(bottom_left, in) || !read(top_right, in)) {
		response = mpi_response::SERVER_PARSE_MESSAGE_ERROR;
		success = false;
	} else {
		if (sim.get_map(bottom_left, top_right, patches)) {
			response = mpi_response::SUCCESS;
		} else {
			for (auto entry : patches)
				free(entry.value);
			patches.clear();
			response = mpi_response::FAILURE;
		}
	}

	default_scribe scribe;
	memory_stream mem_stream = memory_stream(sizeof(message_type) + sizeof(response) + sizeof(hash_map<position, patch_state>));
	fixed_width_stream<memory_stream> out(mem_stream);
	success &= write(message_type::GET_MAP_RESPONSE, out) && write(response, out)
			&& (response != mpi_response::SUCCESS || write(patches, out, scribe, sim.get_config()))
			&& send_message(connection, mem_stream.buffer, mem_stream.position);
	for (auto entry : patches)
		free(entry.value);
	return success;
}

template<typename Stream, typename SimulatorData>
inline bool receive_set_active(
		Stream& in, socket_type& connection,
		const array<uint64_t>& agent_ids,
		simulator<SimulatorData>& sim)
{
	uint64_t agent_id = UINT64_MAX;
	bool active, success = true;
	mpi_response response;
	if (!read(agent_id, in) || !read(active, in)) {
		response = mpi_response::SERVER_PARSE_MESSAGE_ERROR;
		success = false;
	} else {
		if (!agent_ids.contains(agent_id)) {
			response = mpi_response::INVALID_AGENT_ID;
		} else {
			sim.set_agent_active(agent_id, active);
			response = mpi_response::SUCCESS;
		}
	}
	memory_stream mem_stream = memory_stream(sizeof(message_type) + sizeof(agent_id) + sizeof(response));
	fixed_width_stream<memory_stream> out(mem_stream);

	success &= write(message_type::SET_ACTIVE_RESPONSE, out)
			&& write(agent_id, out) && write(response, out)
			&& send_message(connection, mem_stream.buffer, mem_stream.position);
	return success;
}

template<typename Stream, typename SimulatorData>
inline bool receive_is_active(
		Stream& in, socket_type& connection,
		const array<uint64_t>& agent_ids,
		simulator<SimulatorData>& sim)
{
	uint64_t agent_id = UINT64_MAX;
	bool success = true;
	mpi_response response;
	if (!read(agent_id, in)) {
		response = mpi_response::SERVER_PARSE_MESSAGE_ERROR;
		success = false;
	} else {
		if (!agent_ids.contains(agent_id)) {
			response = mpi_response::INVALID_AGENT_ID;
		} else if (sim.is_agent_active(agent_id)) {
			response = mpi_response::SUCCESS;
		} else {
			response = mpi_response::FAILURE;
		}
	}
	memory_stream mem_stream = memory_stream(sizeof(message_type) + sizeof(agent_id) + sizeof(response));
	fixed_width_stream<memory_stream> out(mem_stream);

	success &= write(message_type::IS_ACTIVE_RESPONSE, out)
			&& write(agent_id, out) && write(response, out)
			&& send_message(connection, mem_stream.buffer, mem_stream.position);
	return success;
}

template<typename SimulatorData>
void server_process_message(socket_type& connection,
		hash_map<socket_type, client_info>& connections,
		std::mutex& connection_set_lock,
		simulator<SimulatorData>& sim, server_state& state)
{
	message_type type;
	fixed_width_stream<socket_type> in(connection);
	connection_set_lock.lock();
	uint64_t client_id = connections.get(connection).id;
	connection_set_lock.unlock();
	if (!read(type, in)) return;
	switch (type) {
		case message_type::ADD_AGENT:
			receive_add_agent(in, connection, state.agent_ids.get(client_id), sim); return;
		case message_type::MOVE:
			receive_move(in, connection, state.agent_ids.get(client_id), sim); return;
		case message_type::TURN:
			receive_turn(in, connection, state.agent_ids.get(client_id), sim); return;
		case message_type::DO_NOTHING:
			receive_do_nothing(in, connection, state.agent_ids.get(client_id), sim); return;
		case message_type::GET_MAP:
			receive_get_map(in, connection, sim); return;
		case message_type::SET_ACTIVE:
			receive_set_active(in, connection, state.agent_ids.get(client_id), sim); return;
		case message_type::IS_ACTIVE:
			receive_is_active(in, connection, state.agent_ids.get(client_id), sim); return;

		case message_type::ADD_AGENT_RESPONSE:
		case message_type::MOVE_RESPONSE:
		case message_type::TURN_RESPONSE:
		case message_type::DO_NOTHING_RESPONSE:
		case message_type::GET_MAP_RESPONSE:
		case message_type::SET_ACTIVE_RESPONSE:
		case message_type::IS_ACTIVE_RESPONSE:
		case message_type::STEP_RESPONSE:
			break;
	}
	fprintf(stderr, "server_process_message WARNING: Received message with unrecognized type.\n");
}

template<typename SimulatorData>
inline bool process_new_connection(
		socket_type& connection, client_info& new_client,
		simulator<SimulatorData>& sim, server_state& state)
{
	/* read the client ID or `NEW_CLIENT_REQUEST` */
	uint64_t client_id;
	fixed_width_stream<socket_type> in(connection);
	if (!read(client_id, in)) {
		fprintf(stderr, "process_new_connection ERROR: Failed to read agent_count.\n");
		memory_stream mem_stream = memory_stream(sizeof(mpi_response));
		fixed_width_stream<memory_stream> out(mem_stream);
		write(mpi_response::SERVER_PARSE_MESSAGE_ERROR, out);
		send_message(connection, mem_stream.buffer, mem_stream.position);
		return false;
	}

	if (client_id == NEW_CLIENT_REQUEST) {
		if (!state.agent_ids.check_size()) {
			memory_stream mem_stream = memory_stream(sizeof(mpi_response));
			fixed_width_stream<memory_stream> out(mem_stream);
			write(mpi_response::SERVER_PARSE_MESSAGE_ERROR, out);
			send_message(connection, mem_stream.buffer, mem_stream.position);
			return false;
		}

		bool contains; unsigned int bucket;
		new_client.id = state.agent_ids.table.size + 1;
		array<uint64_t>& agent_ids = state.agent_ids.get(new_client.id, contains, bucket);
#if !defined(NDEBUG)
		if (contains)
			fprintf(stderr, "process_new_connection WARNING: `new_client.id` already exists in `state.agent_ids`.\n");
#endif
		if (!array_init(agent_ids, 8)) {
			memory_stream mem_stream = memory_stream(sizeof(mpi_response));
			fixed_width_stream<memory_stream> out(mem_stream);
			write(mpi_response::SERVER_PARSE_MESSAGE_ERROR, out);
			send_message(connection, mem_stream.buffer, mem_stream.position);
			return false;
		}
		state.agent_ids.table.keys[bucket] = new_client.id;
		state.agent_ids.table.size++;

		/* respond to the client */
		memory_stream mem_stream = memory_stream(sizeof(mpi_response) + sizeof(uint64_t) + sizeof(sim.time) + sizeof(simulator_config));
		fixed_width_stream<memory_stream> out(mem_stream);
		const simulator_config& config = sim.get_config();
		return write(mpi_response::SUCCESS, out)
			&& write(sim.time, out) && write(config, out)
			&& write(new_client.id, out)
			&& send_message(connection, mem_stream.buffer, mem_stream.position);

	} else {
		/* first check if the requested client ID exists */
		bool contains;
		const array<uint64_t>& agent_ids = state.agent_ids.get(client_id, contains);
		if (!contains) {
			memory_stream mem_stream = memory_stream(sizeof(mpi_response));
			fixed_width_stream<memory_stream> out(mem_stream);
			write(mpi_response::INVALID_AGENT_ID, out);
			send_message(connection, mem_stream.buffer, mem_stream.position);
			return false;
		}
		new_client.id = client_id;

		/* respond to the client */
		memory_stream mem_stream = memory_stream(sizeof(mpi_response) + sizeof(unsigned int) + sizeof(sim.time) + sizeof(simulator_config));
		fixed_width_stream<memory_stream> out(mem_stream);
		const simulator_config& config = sim.get_config();
		if (!write(mpi_response::SUCCESS, out)
		 || !write(sim.time, out) || !write(config, out)
		 || !write((unsigned int) agent_ids.length, out))
		{
			fprintf(stderr, "process_new_connection ERROR: Error sending simulation time and configuration.\n");
			return false;
		}

		if (agent_ids.length > 0) {
			agent_state** agent_states = (agent_state**) malloc(sizeof(agent_state*) * agent_ids.length);
			sim.get_agent_states(agent_states, agent_ids.data, (unsigned int) agent_ids.length);

			/* send the requested agent states to the client */
			for (unsigned int i = 0; i < agent_ids.length; i++) {
				std::unique_lock<std::mutex> lock(agent_states[i]->lock);
				if (!write(*agent_states[i], out, config)) {
					free(agent_states);
					return false;
				}
			}
			free(agent_states);
		}

		return send_message(connection, mem_stream.buffer, mem_stream.position);
	}
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
		const array<uint64_t>& agent_ids = server.state.agent_ids.get(client_connection.value.id);
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
bool init_server(async_server& new_server, simulator<SimulatorData>& sim,
		uint16_t server_port, unsigned int connection_queue_capacity, unsigned int worker_count)
{
	std::condition_variable cv; std::mutex lock;
	auto dispatch = [&]() {
		run_server(new_server.server_socket, server_port,
				connection_queue_capacity, worker_count, new_server.status, cv, lock,
				new_server.client_connections, new_server.connection_set_lock,
				server_process_message<SimulatorData>, process_new_connection<SimulatorData>, sim, new_server.state);
	};
	new_server.status = server_status::STARTING;
	new_server.server_thread = std::thread(dispatch);

	std::unique_lock<std::mutex> lck(lock);
	while (new_server.status == server_status::STARTING)
		cv.wait(lck);
	lck.unlock();
	if (new_server.status == server_status::STOPPING && new_server.server_thread.joinable()) {
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
 * \param new_server The sync_server structure containing the state of the new
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
inline bool init_server(sync_server& new_server, simulator<SimulatorData>& sim,
		uint16_t server_port, unsigned int connection_queue_capacity, unsigned int worker_count)
{
	socket_type server_socket;
	server_status dummy = server_status::STARTING;
	std::condition_variable cv; std::mutex lock;
	return run_server(
			server_socket, server_port, connection_queue_capacity, worker_count, dummy, cv, lock,
			new_server.client_connections, new_server.connection_set_lock,
			server_process_message<SimulatorData>, process_new_connection<SimulatorData>, sim, new_server.state);
}

/**
 * Shuts down the asynchronous server given by `server`.
 */
void stop_server(async_server& server) {
	server.status = server_status::STOPPING;
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
 * `on_add_agent(ClientType&, uint64_t, mpi_response, agent_state&)` will be
 * invoked, where the first argument is `c`, the second is the ID of the new
 * agent (which will be UINT64_MAX upon error), and the third is the response
 * (SUCCESS if successful, and a different value if an error occurred), and 
 * the fourth is the state of the new agent. Note the fourth argument is
 * uninitialized if an error occurred.
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
 * responds, the function `on_move(ClientType&, uint64_t, mpi_response)` will
 * be invoked, where the first argument is `c`, the second is `agent_id`, and
 * the third is the response: SUCCESS if successful, and a different value if
 * an error occurred.
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
 * responds, the function `on_turn(ClientType&, uint64_t, mpi_response)` will
 * be invoked, where the first argument is `c`, the second is `agent_id`, and
 * the third is the response: SUCCESS if successful, and a different value if
 * an error occurred.
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
 * Sends a `do_nothing` message to the server from the client `c`. Once the
 * server responds, the function
 * `on_do_nothing(ClientType&, uint64_t, mpi_response)` will be invoked, where
 * the first argument is `c`, the second is `agent_id`, and the third is the
 * response: SUCCESS if successful, and a different value if an error occurred.
 *
 * \returns `true` if the sending is successful; `false` otherwise.
 */
template<typename ClientType>
bool send_do_nothing(ClientType& c, uint64_t agent_id) {
	memory_stream mem_stream = memory_stream(sizeof(message_type) + sizeof(agent_id));
	fixed_width_stream<memory_stream> out(mem_stream);
	return write(message_type::DO_NOTHING, out)
		&& write(agent_id, out)
		&& send_message(c.connection, mem_stream.buffer, mem_stream.position);
}

/**
 * Sends an `get_map` message to the server from the client `c`. Once the
 * server responds, the function
 * `on_get_map(ClientType&, mpi_response, hash_map<position, patch_state>*)`
 * will be invoked, where the first argument is `c`, and the second is the
 * response (SUCCESS if successful, and a different value if an error occurred),
 * and the third is a pointer to a map containing the state information of the
 * retrieved patches. The third argument is uninitialized if the mpi_response
 * is not SUCCESS. Memory ownership of the hash_map is passed to `on_get_map`.
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

/**
 * Sends an `set_active` message to the server from the client `c`. Once the
 * server responds, the function
 * `on_set_active(ClientType&, uint64_t, mpi_response)` will be invoked, where
 * the first argument is `c`, the second is `agent_id`, and the third is the
 * response: SUCCESS if successful, and a different value if an error occurred.
 *
 * \returns `true` if the sending is successful; `false` otherwise.
 */
template<typename ClientType>
bool send_set_active(ClientType& c, uint64_t agent_id, bool active) {
	memory_stream mem_stream = memory_stream(sizeof(message_type) + sizeof(agent_id) + sizeof(active));
	fixed_width_stream<memory_stream> out(mem_stream);
	return write(message_type::SET_ACTIVE, out)
		&& write(agent_id, out) && write(active, out)
		&& send_message(c.connection, mem_stream.buffer, mem_stream.position);
}

/**
 * Sends an `is_active` message to the server from the client `c`. Once the
 * server responds, the function
 * `on_is_active(ClientType&, uint64_t, mpi_response)` will be invoked, where
 * the first argument is `c`, and the second is `agent_id`, and the third is
 * the response: SUCCESS if active, FAILURE if inactive, or a different value
 * if an error occurred.
 *
 * \returns `true` if the sending is successful; `false` otherwise.
 */
template<typename ClientType>
bool send_is_active(ClientType& c, uint64_t agent_id) {
	memory_stream mem_stream = memory_stream(sizeof(message_type) + sizeof(agent_id));
	fixed_width_stream<memory_stream> out(mem_stream);
	return write(message_type::IS_ACTIVE, out)
		&& write(agent_id, out)
		&& send_message(c.connection, mem_stream.buffer, mem_stream.position);
}

template<typename ClientType>
inline bool receive_add_agent_response(ClientType& c) {
	mpi_response response;
	uint64_t agent_id = UINT64_MAX;
	bool success = true;
	agent_state& state = *((agent_state*) alloca(sizeof(agent_state)));
	fixed_width_stream<socket_type> in(c.connection);
	if (!read(agent_id, in)) {
		response = mpi_response::CLIENT_PARSE_MESSAGE_ERROR;
		success = false;
	} else if (agent_id == UINT64_MAX) {
		response = mpi_response::FAILURE;
	} else if (!read(state, in, c.config)) {
		response = mpi_response::CLIENT_PARSE_MESSAGE_ERROR;
		success = false;
	} else {
		response = mpi_response::SUCCESS;
	}
	on_add_agent(c, agent_id, response, state);
	if (agent_id != UINT64_MAX) free(state);
	return success;
}

template<typename ClientType>
inline bool receive_move_response(ClientType& c) {
	mpi_response response;
	uint64_t agent_id = 0;
	bool success = true;
	fixed_width_stream<socket_type> in(c.connection);
	if (!read(agent_id, in) || !read(response, in)) {
		response = mpi_response::CLIENT_PARSE_MESSAGE_ERROR;
		success = false;
	}
	on_move(c, agent_id, response);
	return success;
}

template<typename ClientType>
inline bool receive_turn_response(ClientType& c) {
	mpi_response response;
	uint64_t agent_id = 0;
	bool success = true;
	fixed_width_stream<socket_type> in(c.connection);
	if (!read(agent_id, in) || !read(response, in)) {
		response = mpi_response::CLIENT_PARSE_MESSAGE_ERROR;
		success = false;
	}
	on_turn(c, agent_id, response);
	return success;
}

template<typename ClientType>
inline bool receive_do_nothing_response(ClientType& c) {
	mpi_response response;
	uint64_t agent_id = 0;
	bool success = true;
	fixed_width_stream<socket_type> in(c.connection);
	if (!read(agent_id, in) || !read(response, in)) {
		response = mpi_response::CLIENT_PARSE_MESSAGE_ERROR;
		success = false;
	}
	on_do_nothing(c, agent_id, response);
	return success;
}

template<typename ClientType>
inline bool receive_get_map_response(ClientType& c) {
	mpi_response response;
	default_scribe scribe;
	bool success = true;
	hash_map<position, patch_state>* patches = NULL;
	fixed_width_stream<socket_type> in(c.connection);
	if (!read(response, in)) {
		response = mpi_response::CLIENT_PARSE_MESSAGE_ERROR;
		success = false;
	} else if (response == mpi_response::SUCCESS) {
		patches = (hash_map<position, patch_state>*) malloc(sizeof(hash_map<position, patch_state>));
		if (patches == NULL) {
			fprintf(stderr, "receive_get_map_response ERROR: Out of memory.\n");
			response = mpi_response::CLIENT_PARSE_MESSAGE_ERROR;
			success = false;
		} else if (!read(*patches, in, alloc_position_keys, scribe, c.config)) {
			response = mpi_response::CLIENT_PARSE_MESSAGE_ERROR;
			free(patches); success = false;
		}
	}
	/* ownership of `patches` is passed to the callee */
	on_get_map(c, response, patches);
	return success;
}

template<typename ClientType>
inline bool receive_set_active_response(ClientType& c) {
	mpi_response response;
	uint64_t agent_id = 0;
	bool success = true;
	fixed_width_stream<socket_type> in(c.connection);
	if (!read(agent_id, in) || !read(response, in)) {
		response = mpi_response::CLIENT_PARSE_MESSAGE_ERROR;
		success = false;
	}
	on_set_active(c, agent_id, response);
	return success;
}

template<typename ClientType>
inline bool receive_is_active_response(ClientType& c) {
	mpi_response response;
	uint64_t agent_id = 0;
	bool success = true;
	fixed_width_stream<socket_type> in(c.connection);
	if (!read(agent_id, in) || !read(response, in)) {
		response = mpi_response::CLIENT_PARSE_MESSAGE_ERROR;
		success = false;
	}
	on_is_active(c, agent_id, response);
	return success;
}

template<typename ClientType>
inline bool receive_step_response(ClientType& c) {
	bool success = true;
	mpi_response response = mpi_response::SUCCESS;
	array<uint64_t>& agent_ids = *((array<uint64_t>*) alloca(sizeof(array<uint64_t>)));

	fixed_width_stream<socket_type> in(c.connection);
	agent_state* agents = NULL;
	if (!read(agent_ids, in)) {
		response = mpi_response::CLIENT_PARSE_MESSAGE_ERROR;
		agent_ids.data = NULL; success = false;
	} else {
		agents = (agent_state*) malloc(sizeof(agent_state) * agent_ids.length);
		if (agents == NULL) {
			fprintf(stderr, "receive_step_response ERROR: Out of memory.\n");
			response = mpi_response::CLIENT_PARSE_MESSAGE_ERROR;
			free(agent_ids); success = false;
		} else {
			for (unsigned int i = 0; i < agent_ids.length; i++) {
				if (!read(agents[i], in, c.config)) {
					for (unsigned int j = 0; j < i; j++) free(agents[j]);
					response = mpi_response::CLIENT_PARSE_MESSAGE_ERROR;
					free(agents); free(agent_ids); agents = NULL;
					success = false; break;
				}
			}
		}
	}

	on_step(c, response, (const array<uint64_t>&) agent_ids, (const agent_state*) agents);
	if (agent_ids.data != NULL) {
		for (unsigned int i = 0; i < agent_ids.length; i++)
			free(agents[i]);
		free(agent_ids);
	}
	if (agents != NULL) free(agents);
	return success;
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
		case message_type::DO_NOTHING_RESPONSE:
			receive_do_nothing_response(c); continue;
		case message_type::GET_MAP_RESPONSE:
			receive_get_map_response(c); continue;
		case message_type::SET_ACTIVE_RESPONSE:
			receive_set_active_response(c); continue;
		case message_type::IS_ACTIVE_RESPONSE:
			receive_is_active_response(c); continue;
		case message_type::STEP_RESPONSE:
			receive_step_response(c); continue;

		case message_type::ADD_AGENT:
		case message_type::MOVE:
		case message_type::TURN:
		case message_type::DO_NOTHING:
		case message_type::GET_MAP:
		case message_type::SET_ACTIVE:
		case message_type::IS_ACTIVE:
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
 * \returns The simulator time if successful; `UINT64_MAX` otherwise.
 */
template<typename ClientData>
uint64_t connect_client(client<ClientData>& new_client,
		const char* server_address, const char* server_port,
		uint64_t& client_id)
{
	uint64_t simulator_time;
	auto process_connection = [&](socket_type& connection)
	{
		new_client.connection = connection;

		/* send the client ID */
		memory_stream mem_stream = memory_stream(sizeof(uint64_t));
		fixed_width_stream<memory_stream> out(mem_stream);
		if (!write(NEW_CLIENT_REQUEST, out)
		 || !send_message(connection, mem_stream.buffer, mem_stream.position))
		{
			fprintf(stderr, "connect_client ERROR: Error connecting new client.\n");
			stop_client(new_client); return false;
		}

		/* read the simulator response */
		mpi_response response;
		fixed_width_stream<socket_type> in(connection);
		if (!read(response, in)) {
			fprintf(stderr, "connect_client ERROR: Error receiving response from server.\n");
			stop_client(new_client); return false;
		}

		/* read the simulator time and configuration */
		simulator_config& config = *((simulator_config*) alloca(sizeof(simulator_config)));
		if (!read(simulator_time, in)
		 || !read(config, in)
		 || !read(client_id, in))
		{
			fprintf(stderr, "connect_client ERROR: Error receiving simulator time and configuration.\n");
			stop_client(new_client); return false;
		}
		swap(new_client.config, config);
		free(config); /* free the old configuration */

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
 * Attempts to connect the given client `existing_client` with ID given by
 * `client_id` to the server at `server_address:server_port`. A separate thread
 * will be dispatched to listen for responses from the server. `stop_client`
 * should be used to disconnect from the server and stop the listener thread.
 *
 * \param existing_client The client with which to attempt the connection.
 * \param client_id The ID of the client.
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
uint64_t reconnect_client(
		client<ClientData>& existing_client, uint64_t client_id,
		const char* server_address, const char* server_port,
		uint64_t*& agent_ids, agent_state*& agent_states,
		unsigned int& agent_count)
{
	uint64_t simulator_time;
	auto process_connection = [&](socket_type& connection)
	{
		existing_client.connection = connection;

		/* send the client ID */
		memory_stream mem_stream = memory_stream(sizeof(uint64_t));
		fixed_width_stream<memory_stream> out(mem_stream);
		if (!write(client_id, out)
		 || !send_message(connection, mem_stream.buffer, mem_stream.position))
		{
			fprintf(stderr, "reconnect_client ERROR: Error requesting agent states.\n");
			stop_client(existing_client); return false;
		}

		/* read the simulator response */
		mpi_response response;
		fixed_width_stream<socket_type> in(connection);
		if (!read(response, in)) {
			fprintf(stderr, "reconnect_client ERROR: Error receiving response from server.\n");
			stop_client(existing_client); return false;
		}

		/* read the simulator time and configuration */
		simulator_config& config = *((simulator_config*) alloca(sizeof(simulator_config)));
		if (!read(simulator_time, in)
		 || !read(config, in)
		 || !read(agent_count, in))
		{
			fprintf(stderr, "reconnect_client ERROR: Error receiving simulator time and configuration.\n");
			stop_client(existing_client); return false;
		}
		swap(existing_client.config, config);
		free(config); /* free the old configuration */

		agent_ids = NULL;
		agent_ids = (uint64_t*) malloc(max((size_t) 1, sizeof(uint64_t) * agent_count));
		agent_states = (agent_state*) malloc(max((size_t) 1, sizeof(agent_state) * agent_count));
		if (agent_ids == NULL || agent_states == NULL) {
			if (agent_ids != NULL) free(agent_ids);
			fprintf(stderr, "reconnect_client ERROR: Out of memory.\n");
			stop_client(existing_client); return false;
		}

		/* read the agent IDs associated with this client */
		if (!read(agent_ids, in, agent_count)) {
			fprintf(stderr, "reconnect_client ERROR: Error reading agent IDs.\n");
			free(agent_ids); free(agent_states);
			stop_client(existing_client); return false;
		}

		/* read the agent states for the requested agent IDs */
		for (unsigned int i = 0; i < agent_count; i++) {
			if (!read(agent_states[i], in, existing_client.config)) {
				for (unsigned int j = 0; j < i; j++) free(agent_states[j]);
				free(agent_ids); free(agent_states);
				stop_client(existing_client); return false;
			}
		}

		auto dispatch = [&]() {
			run_response_listener(existing_client);
		};
		existing_client.response_listener = std::thread(dispatch);
		return true;
	};

	existing_client.client_running = true;
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
uint64_t connect_client(client<ClientData>& new_client,
		const char* server_address, uint16_t server_port,
		uint64_t& client_id)
{
	constexpr static unsigned int BUFFER_SIZE = 8;
	char port_str[BUFFER_SIZE];
	if (snprintf(port_str, BUFFER_SIZE, "%u", server_port) > (int) BUFFER_SIZE - 1)
		return false;

	return connect_client(new_client, server_address, port_str, client_id);
}

/**
 * Attempts to connect the given client `existing_client` with ID given by
 * `client_id` to the server at `server_address:server_port`. A separate thread
 * will be dispatched to listen for responses from the server. `stop_client`
 * should be used to disconnect from the server and stop the listener thread.
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
uint64_t reconnect_client(
		client<ClientData>& existing_client, uint64_t client_id,
		const char* server_address, uint16_t server_port,
		uint64_t*& agent_ids, agent_state*& agent_states,
		unsigned int& agent_count)
{
	constexpr static unsigned int BUFFER_SIZE = 8;
	char port_str[BUFFER_SIZE];
	if (snprintf(port_str, BUFFER_SIZE, "%u", server_port) > (int) BUFFER_SIZE - 1)
		return false;

	return reconnect_client(existing_client, client_id, server_address, port_str, agent_ids, agent_states, agent_count);
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

} /* namespace jbw */

#endif /* JBW_MPI_H_ */
