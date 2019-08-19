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
	REMOVE_AGENT,
	REMOVE_AGENT_RESPONSE,
	REMOVE_CLIENT,
	ADD_SEMAPHORE,
	ADD_SEMAPHORE_RESPONSE,
	REMOVE_SEMAPHORE,
	REMOVE_SEMAPHORE_RESPONSE,
	SIGNAL_SEMAPHORE,
	SIGNAL_SEMAPHORE_RESPONSE,
	GET_SEMAPHORES,
	GET_SEMAPHORES_RESPONSE,
	MOVE,
	MOVE_RESPONSE,
	TURN,
	TURN_RESPONSE,
	DO_NOTHING,
	DO_NOTHING_RESPONSE,
	GET_MAP,
	GET_MAP_RESPONSE,
	GET_AGENT_IDS,
	GET_AGENT_IDS_RESPONSE,
	GET_AGENT_STATES,
	GET_AGENT_STATES_RESPONSE,
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
	case message_type::REMOVE_AGENT:     return core::print("REMOVE_AGENT", out);
	case message_type::REMOVE_CLIENT:    return core::print("REMOVE_CLIENT", out);
	case message_type::ADD_SEMAPHORE:    return core::print("ADD_SEMAPHORE", out);
	case message_type::REMOVE_SEMAPHORE: return core::print("REMOVE_SEMAPHORE", out);
	case message_type::SIGNAL_SEMAPHORE: return core::print("SIGNAL_SEMAPHORE", out);
	case message_type::GET_SEMAPHORES:   return core::print("GET_SEMAPHORES", out);
	case message_type::MOVE:             return core::print("MOVE", out);
	case message_type::TURN:             return core::print("TURN", out);
	case message_type::DO_NOTHING:       return core::print("DO_NOTHING", out);
	case message_type::GET_MAP:          return core::print("GET_MAP", out);
	case message_type::GET_AGENT_IDS:    return core::print("GET_AGENT_IDS", out);
	case message_type::GET_AGENT_STATES: return core::print("GET_AGENT_STATES", out);
	case message_type::SET_ACTIVE:       return core::print("SET_ACTIVE", out);
	case message_type::IS_ACTIVE:        return core::print("IS_ACTIVE", out);

	case message_type::ADD_AGENT_RESPONSE:        return core::print("ADD_AGENT_RESPONSE", out);
	case message_type::REMOVE_AGENT_RESPONSE:     return core::print("REMOVE_AGENT_RESPONSE", out);
	case message_type::ADD_SEMAPHORE_RESPONSE:    return core::print("ADD_SEMAPHORE_RESPONSE", out);
	case message_type::REMOVE_SEMAPHORE_RESPONSE: return core::print("REMOVE_SEMAPHORE_RESPONSE", out);
	case message_type::SIGNAL_SEMAPHORE_RESPONSE: return core::print("SIGNAL_SEMAPHORE_RESPONSE", out);
	case message_type::GET_SEMAPHORES_RESPONSE:   return core::print("GET_SEMAPHORES_RESPONSE", out);
	case message_type::MOVE_RESPONSE:             return core::print("MOVE_RESPONSE", out);
	case message_type::TURN_RESPONSE:             return core::print("TURN_RESPONSE", out);
	case message_type::DO_NOTHING_RESPONSE:       return core::print("DO_NOTHING_RESPONSE", out);
	case message_type::GET_MAP_RESPONSE:          return core::print("GET_MAP_RESPONSE", out);
	case message_type::GET_AGENT_IDS_RESPONSE:    return core::print("GET_AGENT_IDS_RESPONSE", out);
	case message_type::GET_AGENT_STATES_RESPONSE: return core::print("GET_AGENT_STATES_RESPONSE", out);
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

struct permissions {
	bool add_agent;
	bool remove_agent;
	bool remove_client;
	bool set_active;
	bool get_map;
	bool get_agent_ids;
	bool get_agent_states;
	bool manage_semaphores;
	bool get_semaphores;

	permissions() { }
	constexpr permissions(bool value) :
		add_agent(value), remove_agent(value), remove_client(value),
		set_active(value), get_map(value), get_agent_ids(value),
		get_agent_states(value), manage_semaphores(value),
		get_semaphores(value)
	{ }

	static inline void swap(permissions& first, permissions& second) {
		core::swap(first.add_agent, second.add_agent);
		core::swap(first.remove_agent, second.remove_agent);
		core::swap(first.remove_client, second.remove_client);
		core::swap(first.set_active, second.set_active);
		core::swap(first.get_map, second.get_map);
		core::swap(first.get_agent_ids, second.get_agent_ids);
		core::swap(first.get_agent_states, second.get_agent_states);
		core::swap(first.manage_semaphores, second.manage_semaphores);
		core::swap(first.get_semaphores, second.get_semaphores);
	}

	static constexpr permissions grant_all() { return permissions(true); }
	static constexpr permissions deny_all() { return permissions(false); }
};

template<typename Stream>
bool read(permissions& perms, Stream& in) {
	return read(perms.add_agent, in)
		&& read(perms.remove_agent, in)
		&& read(perms.remove_client, in)
		&& read(perms.set_active, in)
		&& read(perms.get_map, in)
		&& read(perms.get_agent_ids, in)
		&& read(perms.get_agent_states, in)
		&& read(perms.manage_semaphores, in)
		&& read(perms.get_semaphores, in);
}

template<typename Stream>
bool write(const permissions& perms, Stream& out) {
	return write(perms.add_agent, out)
		&& write(perms.remove_agent, out)
		&& write(perms.remove_client, out)
		&& write(perms.set_active, out)
		&& write(perms.get_map, out)
		&& write(perms.get_agent_ids, out)
		&& write(perms.get_agent_states, out)
		&& write(perms.manage_semaphores, out)
		&& write(perms.get_semaphores, out);
}

struct client_state {
	std::mutex lock;
	array<uint64_t> agent_ids;
	array<uint64_t> semaphore_ids;
	permissions perms;

	static inline void free(client_state& cstate) {
		cstate.lock.~mutex();
		core::free(cstate.agent_ids);
		core::free(cstate.semaphore_ids);
	}
};

inline bool init(client_state& cstate, const permissions& perms) {
	cstate.perms = perms;
	if (!array_init(cstate.agent_ids, 8)) {
		return false;
	} else if (!array_init(cstate.semaphore_ids, 4)) {
		free(cstate.agent_ids);
		return false;
	}
	new (&cstate.lock) std::mutex();
	return true;
}

template<typename Stream>
bool read(client_state& cstate, Stream& in) {
	if (!read(cstate.perms, in)
	 || !read(cstate.agent_ids, in)
	 || !read(cstate.semaphore_ids, in)) return false;
	new (&cstate.lock) std::mutex();
	return true;
}

template<typename Stream>
bool write(const client_state& cstate, Stream& out) {
	return write(cstate.perms, out)
		&& write(cstate.agent_ids, out)
		&& write(cstate.semaphore_ids, out);
}

/**
 * A structure that keeps track of additional state for the MPI server.
 */
struct server_state {
	std::mutex client_states_lock;
	hash_map<uint64_t, client_state*> client_states;
	permissions default_client_permissions;
	uint64_t client_id_counter;

	server_state() : client_states(16), client_id_counter(1) { default_client_permissions = { 0 }; }
	~server_state() { free_helper(); }

	static inline void swap(server_state& first, server_state& second) {
		core::swap(first.client_states, second.client_states);
		core::swap(first.default_client_permissions, second.default_client_permissions);
		core::swap(first.client_id_counter, second.client_id_counter);
	}

	static inline void free(server_state& state) {
		state.free_helper();
		core::free(state.client_states);
		state.client_states_lock.~mutex();
	}

private:
	inline void free_helper() {
		for (auto entry : client_states) {
			core::free(*entry.value);
			core::free(entry.value);
		}
	}
};

bool init(server_state& state) {
	state.client_id_counter = 1;
	state.default_client_permissions = { 0 };
	new (&state.client_states_lock) std::mutex();
	return hash_map_init(state.client_states, 16);
}

template<typename Stream>
bool read(server_state& state, Stream& in) {
	unsigned int client_state_count;
	if (!read(state.client_id_counter, in)
	 || !read(state.default_client_permissions, in)
	 || !read(client_state_count, in)) return false;

	if (!hash_map_init(state.client_states, 1 << (core::log2(RESIZE_THRESHOLD_INVERSE * (client_state_count + 1)) + 1)))
		return false;

	for (unsigned int i = 0; i < client_state_count; i++) {
		uint64_t id;
		client_state* cstate = (client_state*) malloc(sizeof(client_state));
		if (cstate == nullptr || !read(id, in) || !read(*cstate, in)) {
			if (cstate != nullptr) free(cstate);
			for (auto entry : state.client_states) {
				free(*entry.value); free(entry.value);
				return false;
			}
		}

		unsigned bucket = state.client_states.table.index_to_insert(id);
		state.client_states.table.keys[bucket] = id;
		state.client_states.table.size++;
		state.client_states.values[bucket] = cstate;
	}
	new (&state.client_states_lock) std::mutex();
	return true;
}

/* **NOTE:** this function assumes the variables in the simulator are not modified during writing. */
template<typename Stream>
bool write(const server_state& state, Stream& out) {
	if (!write(state.client_id_counter, out)
	 || !write(state.default_client_permissions, out)
	 || !write(state.client_states.table.size, out))
		return false;

	for (const auto& entry : state.client_states) {
		if (!write(entry.key, out) || !write(*entry.value, out))
			return false;
	}
	return true;
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

template<typename ServerType>
inline void set_permissions(ServerType& server,
		uint64_t client_id, const permissions& perms)
{
	bool contains;
	std::unique_lock<std::mutex> lock(server.state.client_states_lock);
	client_state* cstate = server.state.client_states.get(client_id, contains);
	if (!contains) return; /* the client was already destroyed */
	std::unique_lock<std::mutex> cstate_lock(cstate->lock);
	cstate->perms = perms;
}

template<typename ServerType>
inline permissions get_permissions(
		ServerType& server, uint64_t client_id)
{
	bool contains;
	std::unique_lock<std::mutex> lock(server.state.client_states_lock);
	client_state* cstate = server.state.client_states.get(client_id, contains);
	if (!contains) {
		fprintf(stderr, "get_permissions ERROR: The client with the given ID was already removed.\n");
		return server.state.default_client_permissions;
	}
	std::unique_lock<std::mutex> cstate_lock(cstate->lock);
	return cstate->perms;
}

/**
 * Writes the bytes in `data` of length `length` to the TCP socket in `socket`.
 */
inline bool send_message(socket_type& socket, const void* data, unsigned int length) {
	return send(socket.handle, (const char*) data, length, 0) != 0;
}

inline client_state* acquire_client_lock(
		server_state& state, uint64_t client_id)
{
	bool contains;
	state.client_states_lock.lock();
	client_state* cstate = state.client_states.get(client_id, contains);
	if (!contains) {
		/* the client was destroyed so return `nullptr` */
		state.client_states_lock.unlock();
		return nullptr;
	}
	cstate->lock.lock();
	state.client_states_lock.unlock();
	return cstate;
}

/* Precondition: `state.client_states_lock` must be held by the calling thread. */
template<typename Stream, typename SimulatorData>
inline bool receive_add_agent(
		Stream& in, socket_type& connection,
		server_state& state, uint64_t client_id,
		simulator<SimulatorData>& sim)
{
	bool contains;
	client_state* cstate = state.client_states.get(client_id, contains);
	if (!contains) {
		state.client_states_lock.unlock();
		return true; /* the client was already destroyed */
	}
	cstate->lock.lock();
	state.client_states_lock.unlock();

	status response;
	uint64_t new_agent_id = 0; agent_state* new_agent = nullptr;
	if (!cstate->perms.add_agent) {
		/* the client has no permission for this operation */
		response = status::PERMISSION_ERROR;
	} else {
		/* We have to unlock this to avoid deadlock since other simulator
		   functions (i.e. `move`, `turn`, `do_nothing`) can cause the
		   simulator to step. This calls `send_step_response` which needs the
		   client_state locks. */
		cstate->lock.unlock();
		response = sim.add_agent(new_agent_id, new_agent);

		cstate = acquire_client_lock(state, client_id);
		if (cstate == nullptr) {
			/* the client was destroyed while we didn't have the client lock */
			if (response == status::OK)
				sim.remove_agent(new_agent_id);
			return true;
		}
		if (response == status::OK) {
			if (!cstate->agent_ids.add(new_agent_id)) {
				sim.remove_agent(new_agent_id);
				response = status::SERVER_OUT_OF_MEMORY;
			}
		} else if (response == status::OUT_OF_MEMORY) {
			response = status::SERVER_OUT_OF_MEMORY;
		}
	}

	memory_stream mem_stream = memory_stream(sizeof(message_type) + sizeof(response) + sizeof(new_agent_id) + sizeof(*new_agent));
	fixed_width_stream<memory_stream> out(mem_stream);
	if (response == status::OK) {
		bool result = write(message_type::ADD_AGENT_RESPONSE, out)
				   && write(response, out)
				   && write(new_agent_id, out)
				   && write(*new_agent, out, sim.get_config())
				   && send_message(connection, mem_stream.buffer, mem_stream.position);
		cstate->lock.unlock();
		return result;
	} else {
		bool result = write(message_type::ADD_AGENT_RESPONSE, out)
				   && write(response, out)
				   && send_message(connection, mem_stream.buffer, mem_stream.position);
		cstate->lock.unlock();
		return result;
	}
}

/* Precondition: `state.client_states_lock` must be held by the calling thread. */
template<typename Stream, typename SimulatorData>
inline bool receive_remove_agent(
		Stream& in, socket_type& connection,
		server_state& state, uint64_t client_id,
		simulator<SimulatorData>& sim)
{
	bool contains;
	client_state* cstate = state.client_states.get(client_id, contains);
	if (!contains) {
		state.client_states_lock.unlock();
		return true; /* the client was already destroyed */
	}
	cstate->lock.lock();
	state.client_states_lock.unlock();

	uint64_t agent_id = UINT64_MAX;
	status response;
	bool success = true;
	if (!read(agent_id, in)) {
		response = status::SERVER_PARSE_MESSAGE_ERROR;
		success = false;
	} else {
		unsigned int index = cstate->agent_ids.index_of(agent_id);
		if (!cstate->perms.remove_agent) {
			/* the client has no permission for this operation */
			response = status::PERMISSION_ERROR;
		} else if (index == cstate->agent_ids.length) {
			response = status::INVALID_AGENT_ID;
		} else {
			/* We have to unlock this to avoid deadlock since other simulator
			   functions (i.e. `move`, `turn`, `do_nothing`) can cause the
			   simulator to step. This calls `send_step_response` which needs the
			   client_state locks. */
			cstate->lock.unlock();

			response = sim.remove_agent(agent_id);

			cstate = acquire_client_lock(state, client_id);
			if (cstate == nullptr)
				/* the client was destroyed while we didn't have the client lock */
				return true;
			if (response == status::OK) {
				cstate->agent_ids.remove(index);
			} else if (response == status::OUT_OF_MEMORY) {
				response = status::SERVER_OUT_OF_MEMORY;
			}
		}
	}

	memory_stream mem_stream = memory_stream(sizeof(message_type) + sizeof(response));
	fixed_width_stream<memory_stream> out(mem_stream);
	success &= write(message_type::REMOVE_AGENT_RESPONSE, out)
			&& write(agent_id, out) && write(response, out)
			&& send_message(connection, mem_stream.buffer, mem_stream.position);
	cstate->lock.unlock();
	return success;
}

/* Precondition: `state.client_states_lock` must be held by the calling thread. */
template<typename Stream, typename SimulatorData>
bool receive_remove_client(
		Stream& in, socket_type& connection,
		server_state& state, uint64_t client_id,
		simulator<SimulatorData>& sim)
{
	bool contains; unsigned int bucket;
	client_state* cstate = state.client_states.get(client_id, contains, bucket);
	if (!contains) {
		state.client_states_lock.unlock();
		return true; /* the client was already destroyed */
	}
	cstate->lock.lock();
	if (!cstate->perms.remove_client) {
		/* the client has no permission for this operation */
		return false;
	}

	array<uint64_t> agent_ids(1), semaphore_ids(1);
	core::swap(cstate->agent_ids, agent_ids);
	core::swap(cstate->semaphore_ids, semaphore_ids);

	/* We have to unlock this to avoid deadlock removing agents and semaphores
	   can cause the simulator to step, which would call `send_step_response`
	   which needs these locks. */
	state.client_states.remove_at(bucket);
	cstate->lock.unlock();
	free(*cstate); free(cstate);
	shutdown(connection.handle, 2);
	state.client_states_lock.unlock();

	while (agent_ids.length > 0)
		sim.remove_agent(agent_ids[--agent_ids.length]);
	while (semaphore_ids.length > 0)
		sim.remove_semaphore(semaphore_ids[--semaphore_ids.length]);
	return true;
}

/* Precondition: `state.client_states_lock` must be held by the calling thread. */
template<typename Stream, typename SimulatorData>
inline bool receive_add_semaphore(
		Stream& in, socket_type& connection,
		server_state& state, uint64_t client_id,
		simulator<SimulatorData>& sim)
{
	bool contains;
	client_state* cstate = state.client_states.get(client_id, contains);
	if (!contains) {
		state.client_states_lock.unlock();
		return true; /* the client was already destroyed */
	}
	cstate->lock.lock();
	state.client_states_lock.unlock();

	status response;
	uint64_t new_semaphore_id = 0;
	if (!cstate->perms.manage_semaphores) {
		/* the client has no permission for this operation */
		response = status::PERMISSION_ERROR;
	} else {
		/* We have to unlock this to avoid deadlock since other simulator
		   functions (i.e. `move`, `turn`, `do_nothing`) can cause the
		   simulator to step. This calls `send_step_response` which needs the
		   client_state locks. */
		cstate->lock.unlock();
		response = sim.add_semaphore(new_semaphore_id);

		cstate = acquire_client_lock(state, client_id);
		if (cstate == nullptr) {
			/* the client was destroyed while we didn't have the client lock */
			if (response == status::OK)
				sim.remove_semaphore(new_semaphore_id);
			return true;
		}
		if (response == status::OK) {
			if (!cstate->semaphore_ids.add(new_semaphore_id)) {
				sim.remove_semaphore(new_semaphore_id);
				response = status::SERVER_OUT_OF_MEMORY;
			}
		} else if (response == status::OUT_OF_MEMORY) {
			response = status::SERVER_OUT_OF_MEMORY;
		}
	}

	memory_stream mem_stream = memory_stream(sizeof(message_type) + sizeof(response) + sizeof(new_semaphore_id));
	fixed_width_stream<memory_stream> out(mem_stream);
	if (response == status::OK) {
		bool result = write(message_type::ADD_SEMAPHORE_RESPONSE, out)
				   && write(response, out)
				   && write(new_semaphore_id, out)
				   && send_message(connection, mem_stream.buffer, mem_stream.position);
		cstate->lock.unlock();
		return result;
	} else {
		bool result = write(message_type::ADD_SEMAPHORE_RESPONSE, out)
				   && write(response, out)
				   && send_message(connection, mem_stream.buffer, mem_stream.position);
		cstate->lock.unlock();
		return result;
	}
}

/* Precondition: `state.client_states_lock` must be held by the calling thread. */
template<typename Stream, typename SimulatorData>
inline bool receive_remove_semaphore(
		Stream& in, socket_type& connection,
		server_state& state, uint64_t client_id,
		simulator<SimulatorData>& sim)
{
	bool contains;
	client_state* cstate = state.client_states.get(client_id, contains);
	if (!contains) {
		state.client_states_lock.unlock();
		return true; /* the client was already destroyed */
	}
	cstate->lock.lock();
	state.client_states_lock.unlock();

	uint64_t semaphore_id = UINT64_MAX;
	status response;
	bool success = true;
	if (!read(semaphore_id, in)) {
		response = status::SERVER_PARSE_MESSAGE_ERROR;
		success = false;
	} else {
		unsigned int index = cstate->semaphore_ids.index_of(semaphore_id);
		if (!cstate->perms.manage_semaphores) {
			/* the client has no permission for this operation */
			response = status::PERMISSION_ERROR;
		} else if (index == cstate->semaphore_ids.length) {
			response = status::INVALID_SEMAPHORE_ID;
		} else {
			/* We have to unlock this to avoid deadlock since other simulator
			   functions (i.e. `move`, `turn`, `do_nothing`) can cause the
			   simulator to step. This calls `send_step_response` which needs the
			   client_state locks. */
			cstate->lock.unlock();

			response = sim.remove_semaphore(semaphore_id);

			cstate = acquire_client_lock(state, client_id);
			if (cstate == nullptr)
				/* the client was destroyed while we didn't have the client lock */
				return true;
			if (response == status::OK) {
				cstate->semaphore_ids.remove(index);
			} else if (response == status::OUT_OF_MEMORY) {
				response = status::SERVER_OUT_OF_MEMORY;
			}
		}
	}

	memory_stream mem_stream = memory_stream(sizeof(message_type) + sizeof(response));
	fixed_width_stream<memory_stream> out(mem_stream);
	success &= write(message_type::REMOVE_SEMAPHORE_RESPONSE, out)
			&& write(semaphore_id, out) && write(response, out)
			&& send_message(connection, mem_stream.buffer, mem_stream.position);
	cstate->lock.unlock();
	return success;
}

/* Precondition: `state.client_states_lock` must be held by the calling thread. */
template<typename Stream, typename SimulatorData>
inline bool receive_signal_semaphore(
		Stream& in, socket_type& connection,
		server_state& state, uint64_t client_id,
		simulator<SimulatorData>& sim)
{
	bool contains;
	client_state* cstate = state.client_states.get(client_id, contains);
	if (!contains) {
		state.client_states_lock.unlock();
		return true; /* the client was already destroyed */
	}
	cstate->lock.lock();
	state.client_states_lock.unlock();

	uint64_t semaphore_id = UINT64_MAX;
	status response;
	bool success = true;
	if (!read(semaphore_id, in)) {
		response = status::SERVER_PARSE_MESSAGE_ERROR;
		success = false;
	} else {
		if (!cstate->perms.manage_semaphores) {
			/* the client has no permission for this operation */
			response = status::PERMISSION_ERROR;
		} else if (semaphore_id == 0 || !cstate->semaphore_ids.contains(semaphore_id)) {
			response = status::INVALID_SEMAPHORE_ID;
		} else {
			/* We have to unlock this to avoid deadlock since other simulator
			   functions (i.e. `move`, `turn`, `do_nothing`) can cause the
			   simulator to step. This calls `send_step_response` which needs the
			   client_state locks. */
			cstate->lock.unlock();
			cstate = nullptr;

			response = sim.signal_semaphore(semaphore_id);
			if (response == status::OUT_OF_MEMORY)
				response = status::SERVER_OUT_OF_MEMORY;
		}
	}

	memory_stream mem_stream = memory_stream(sizeof(message_type) + sizeof(semaphore_id) + sizeof(response));
	fixed_width_stream<memory_stream> out(mem_stream);

	success &= write(message_type::SIGNAL_SEMAPHORE_RESPONSE, out)
			&& write(semaphore_id, out) && write(response, out);
	if (!success) {
		if (cstate != nullptr)
			cstate->lock.unlock();
		return false;
	}

	if (cstate == nullptr) {
		cstate = acquire_client_lock(state, client_id);
		if (cstate == nullptr)
			/* the client was destroyed while we didn't have the client lock */
			return true;
	}
	success = send_message(connection, mem_stream.buffer, mem_stream.position);
	cstate->lock.unlock();
	return success;
}

/* Precondition: `state.client_states_lock` must be held by the calling thread. */
template<typename Stream, typename SimulatorData>
inline bool receive_get_semaphores(
		Stream& in, socket_type& connection,
		server_state& state, uint64_t client_id,
		simulator<SimulatorData>& sim)
{
	bool contains;
	client_state* cstate = state.client_states.get(client_id, contains);
	if (!contains) {
		state.client_states_lock.unlock();
		return true; /* the client was already destroyed */
	}
	cstate->lock.lock();
	state.client_states_lock.unlock();

	status response;
	bool success = true;
	array<pair<uint64_t, bool>> semaphores(64);
	if (!cstate->perms.get_semaphores) {
		/* the client has no permission for this operation */
		response = status::PERMISSION_ERROR;
	} else {
		/* We have to unlock this to avoid deadlock since other simulator
			functions (i.e. `move`, `turn`, `do_nothing`) can cause the
			simulator to step. This calls `send_step_response` which needs the
			client_state locks. */
		cstate->lock.unlock();
		cstate = nullptr;

		response = sim.get_semaphores(semaphores);
		if (response == status::OUT_OF_MEMORY)
			response = status::SERVER_OUT_OF_MEMORY;
	}

	memory_stream mem_stream = memory_stream(sizeof(message_type) + sizeof(response) + sizeof(semaphores.length) + sizeof(pair<uint64_t, bool>));
	fixed_width_stream<memory_stream> out(mem_stream);

	success &= write(message_type::GET_SEMAPHORES_RESPONSE, out)
			&& write(response, out)
			&& write(semaphores.length, out)
			&& write(semaphores.data, out, semaphores.length);
	if (!success) {
		if (cstate != nullptr)
			cstate->lock.unlock();
		return false;
	}

	if (cstate == nullptr) {
		cstate = acquire_client_lock(state, client_id);
		if (cstate == nullptr)
			/* the client was destroyed while we didn't have the client lock */
			return true;
	}
	success = send_message(connection, mem_stream.buffer, mem_stream.position);
	cstate->lock.unlock();
	return success;
}

/* Precondition: `state.client_states_lock` must be held by the calling thread. */
template<typename Stream, typename SimulatorData>
inline bool receive_move(
		Stream& in, socket_type& connection,
		server_state& state, uint64_t client_id,
		simulator<SimulatorData>& sim)
{
	bool contains;
	client_state* cstate = state.client_states.get(client_id, contains);
	if (!contains) {
		state.client_states_lock.unlock();
		return true; /* the client was already destroyed */
	}
	cstate->lock.lock();
	state.client_states_lock.unlock();

	uint64_t agent_id = UINT64_MAX;
	direction dir;
	unsigned int num_steps;
	status response;
	bool success = true;
	if (!read(agent_id, in) || !read(dir, in) || !read(num_steps, in)) {
		response = status::SERVER_PARSE_MESSAGE_ERROR;
		success = false;
	} else {
		if (agent_id == 0 || !cstate->agent_ids.contains(agent_id)) {
			response = status::INVALID_AGENT_ID;
		} else {
			/* We have to unlock this to avoid deadlock since other simulator
			   functions (i.e. `move`, `turn`, `do_nothing`) can cause the
			   simulator to step. This calls `send_step_response` which needs the
			   client_state locks. */
			cstate->lock.unlock();
			cstate = nullptr;

			response = sim.move(agent_id, dir, num_steps);
			if (response == status::OUT_OF_MEMORY)
				response = status::SERVER_OUT_OF_MEMORY;
		}
	}

	memory_stream mem_stream = memory_stream(sizeof(message_type) + sizeof(agent_id) + sizeof(response));
	fixed_width_stream<memory_stream> out(mem_stream);

	success &= write(message_type::MOVE_RESPONSE, out)
			&& write(agent_id, out) && write(response, out);
	if (!success) {
		if (cstate != nullptr)
			cstate->lock.unlock();
		return false;
	}

	if (cstate == nullptr) {
		cstate = acquire_client_lock(state, client_id);
		if (cstate == nullptr)
			/* the client was destroyed while we didn't have the client lock */
			return true;
	}
	success = send_message(connection, mem_stream.buffer, mem_stream.position);
	cstate->lock.unlock();
	return success;
}

/* Precondition: `state.client_states_lock` must be held by the calling thread. */
template<typename Stream, typename SimulatorData>
inline bool receive_turn(
		Stream& in, socket_type& connection,
		server_state& state, uint64_t client_id,
		simulator<SimulatorData>& sim)
{
	bool contains;
	client_state* cstate = state.client_states.get(client_id, contains);
	if (!contains) {
		state.client_states_lock.unlock();
		return true; /* the client was already destroyed */
	}
	cstate->lock.lock();
	state.client_states_lock.unlock();

	uint64_t agent_id = UINT64_MAX;
	direction dir;
	status response;
	bool success = true;
	if (!read(agent_id, in) || !read(dir, in)) {
		response = status::SERVER_PARSE_MESSAGE_ERROR;
		success = false;
	} else {
		if (agent_id == 0 || !cstate->agent_ids.contains(agent_id)) {
			response = status::INVALID_AGENT_ID;
		} else {
			/* We have to unlock this to avoid deadlock since other simulator
			   functions (i.e. `move`, `turn`, `do_nothing`) can cause the
			   simulator to step. This calls `send_step_response` which needs the
			   client_state locks. */
			cstate->lock.unlock();
			cstate = nullptr;

			response = sim.turn(agent_id, dir);
			if (response == status::OUT_OF_MEMORY)
				response = status::SERVER_OUT_OF_MEMORY;
		}
	}

	memory_stream mem_stream = memory_stream(sizeof(message_type) + sizeof(agent_id) + sizeof(response));
	fixed_width_stream<memory_stream> out(mem_stream);

	success &= write(message_type::TURN_RESPONSE, out)
			&& write(agent_id, out) && write(response, out);
	if (!success) {
		if (cstate != nullptr)
			cstate->lock.unlock();
		return false;
	}

	if (cstate == nullptr) {
		cstate = acquire_client_lock(state, client_id);
		if (cstate == nullptr)
			/* the client was destroyed while we didn't have the client lock */
			return true;
	}
	success = send_message(connection, mem_stream.buffer, mem_stream.position);
	cstate->lock.unlock();
	return success;
}

/* Precondition: `state.client_states_lock` must be held by the calling thread. */
template<typename Stream, typename SimulatorData>
inline bool receive_do_nothing(
		Stream& in, socket_type& connection,
		server_state& state, uint64_t client_id,
		simulator<SimulatorData>& sim)
{
	bool contains;
	client_state* cstate = state.client_states.get(client_id, contains);
	if (!contains) {
		state.client_states_lock.unlock();
		return true; /* the client was already destroyed */
	}
	cstate->lock.lock();
	state.client_states_lock.unlock();

	uint64_t agent_id = UINT64_MAX;
	status response;
	bool success = true;
	if (!read(agent_id, in)) {
		response = status::SERVER_PARSE_MESSAGE_ERROR;
		success = false;
	} else {
		if (agent_id == 0 || !cstate->agent_ids.contains(agent_id)) {
			response = status::INVALID_AGENT_ID;
		} else {
			/* We have to unlock this to avoid deadlock since other simulator
			   functions (i.e. `move`, `turn`, `do_nothing`) can cause the
			   simulator to step. This calls `send_step_response` which needs the
			   client_state locks. */
			cstate->lock.unlock();
			cstate = nullptr;

			response = sim.do_nothing(agent_id);
			if (response == status::OUT_OF_MEMORY)
				response = status::SERVER_OUT_OF_MEMORY;
		}
	}
	memory_stream mem_stream = memory_stream(sizeof(message_type) + sizeof(agent_id) + sizeof(response));
	fixed_width_stream<memory_stream> out(mem_stream);

	success &= write(message_type::DO_NOTHING_RESPONSE, out)
			&& write(agent_id, out) && write(response, out);
	if (!success) {
		if (cstate != nullptr)
			cstate->lock.unlock();
		return false;
	}

	if (cstate == nullptr) {
		cstate = acquire_client_lock(state, client_id);
		if (cstate == nullptr)
			/* the client was destroyed while we didn't have the client lock */
			return true;
	}
	success = send_message(connection, mem_stream.buffer, mem_stream.position);
	cstate->lock.unlock();
	return success;
}

/* Precondition: `state.client_states_lock` must be held by the calling thread. */
template<typename Stream, typename SimulatorData>
inline bool receive_get_map(
		Stream& in, socket_type& connection,
		server_state& state, uint64_t client_id,
		simulator<SimulatorData>& sim)
{
	bool contains;
	client_state* cstate = state.client_states.get(client_id, contains);
	if (!contains) {
		state.client_states_lock.unlock();
		return true; /* the client was already destroyed */
	}
	cstate->lock.lock();
	state.client_states_lock.unlock();

	position bottom_left, top_right;
	bool get_scent_map, get_vision_map;
	status response;
	array<array<patch_state>> patches(32);
	bool success = true;
	if (!read(bottom_left, in) || !read(top_right, in) || !read(get_scent_map, in) || !read(get_vision_map, in)) {
		response = status::SERVER_PARSE_MESSAGE_ERROR;
		success = false;
	} else if (!cstate->perms.get_map) {
		/* the client has no permission for this operation */
		response = status::PERMISSION_ERROR;
	} else {
		/* We have to unlock this to avoid deadlock since other simulator
			functions (i.e. `move`, `turn`, `do_nothing`) can cause the
			simulator to step. This calls `send_step_response` which needs the
			client_state locks. */
		cstate->lock.unlock();
		cstate = nullptr;

		if (get_scent_map) {
			if (get_vision_map) {
				response = sim.template get_map<true, true>(bottom_left, top_right, patches);
			} else {
				response = sim.template get_map<true, false>(bottom_left, top_right, patches);
			}
		} else {
			if (get_vision_map) {
				response = sim.template get_map<false, true>(bottom_left, top_right, patches);
			} else {
				response = sim.template get_map<false, false>(bottom_left, top_right, patches);
			}
		}
		if (response != status::OK) {
			for (array<patch_state>& row : patches) {
				for (patch_state& patch : row) free(patch);
				free(row);
			}
			patches.clear();
			if (response == status::OUT_OF_MEMORY)
				response = status::SERVER_OUT_OF_MEMORY;
		}
	}

	memory_stream mem_stream = memory_stream(sizeof(message_type) + sizeof(response) + sizeof(hash_map<position, patch_state>));
	fixed_width_stream<memory_stream> out(mem_stream);
	success &= write(message_type::GET_MAP_RESPONSE, out) && write(response, out)
			&& (response != status::OK || write(patches, out, sim.get_config()));
	if (!success) {
		if (cstate != nullptr)
			cstate->lock.unlock();
		for (array<patch_state>& row : patches) {
			for (patch_state& patch : row) free(patch);
			free(row);
		}
		return false;
	}

	if (cstate == nullptr) {
		cstate = acquire_client_lock(state, client_id);
		if (cstate == nullptr)
			/* the client was destroyed while we didn't have the client lock */
			return true;
	}
	success = send_message(connection, mem_stream.buffer, mem_stream.position);
	cstate->lock.unlock();
	for (array<patch_state>& row : patches) {
		for (patch_state& patch : row) free(patch);
		free(row);
	}
	return success;
}

/* Precondition: `state.client_states_lock` must be held by the calling thread. */
template<typename Stream, typename SimulatorData>
inline bool receive_get_agent_ids(
		Stream& in, socket_type& connection,
		server_state& state, uint64_t client_id,
		simulator<SimulatorData>& sim)
{
	bool contains;
	client_state* cstate = state.client_states.get(client_id, contains);
	if (!contains) {
		state.client_states_lock.unlock();
		return true; /* the client was already destroyed */
	}
	cstate->lock.lock();
	state.client_states_lock.unlock();

	status response;
	array<uint64_t> agent_ids(32);
	bool success = true;
	if (!cstate->perms.get_agent_ids) {
		/* the client has no permission for this operation */
		response = status::PERMISSION_ERROR;
	} else {
		/* We have to unlock this to avoid deadlock since other simulator
			functions (i.e. `move`, `turn`, `do_nothing`) can cause the
			simulator to step. This calls `send_step_response` which needs the
			client_state locks. */
		cstate->lock.unlock();
		cstate = nullptr;

		response = sim.get_agent_ids(agent_ids);
		if (response != status::OK) {
			agent_ids.clear();
			if (response == status::OUT_OF_MEMORY)
				response = status::SERVER_OUT_OF_MEMORY;
		}
	}

	memory_stream mem_stream = memory_stream(sizeof(message_type) + sizeof(response) + sizeof(size_t) + sizeof(uint64_t) * agent_ids.length);
	fixed_width_stream<memory_stream> out(mem_stream);
	success &= write(message_type::GET_AGENT_IDS_RESPONSE, out) && write(response, out)
			&& write(agent_ids.length, out) && write(agent_ids.data, out, agent_ids.length);
	if (!success) {
		if (cstate != nullptr)
			cstate->lock.unlock();
		return false;
	}

	if (cstate == nullptr) {
		cstate = acquire_client_lock(state, client_id);
		if (cstate == nullptr)
			/* the client was destroyed while we didn't have the client lock */
			return true;
	}
	success = send_message(connection, mem_stream.buffer, mem_stream.position);
	cstate->lock.unlock();
	return success;
}

template<typename Stream>
inline bool send_agent_states(
	Stream& out, uint64_t* agent_ids,
	agent_state** agent_states,
	size_t& agent_state_count,
	const simulator_config& config)
{
	/* remove agent IDs from `agent_ids` which don't exist in the simulator */
	size_t old_agent_state_count = agent_state_count;
	unsigned int agent_id_index = 0;
	for (unsigned int i = 0; i < old_agent_state_count; i++) {
		if (agent_states[i] != nullptr)
			agent_ids[agent_id_index++] = agent_ids[i];
	}
	agent_state_count = agent_id_index;

	if (!write(agent_state_count, out)
	 || !write(agent_ids, out, agent_state_count))
	{
		for (unsigned int i = 0; i < old_agent_state_count; i++)
			if (agent_states[i] != nullptr) agent_states[i]->lock.unlock();
		return false;
	}

	/* send the requested agent states to the client */
	for (unsigned int i = 0; i < old_agent_state_count; i++) {
		if (agent_states[i] == nullptr) continue;
		if (!write(*agent_states[i], out, config)) {
			for (unsigned int j = i; j < old_agent_state_count; j++)
				if (agent_states[j] != nullptr) agent_states[j]->lock.unlock();
			return false;
		}
		agent_states[i]->lock.unlock();
	}
	return true;
}

/* Precondition: `state.client_states_lock` must be held by the calling thread. */
template<typename Stream, typename SimulatorData>
inline bool receive_get_agent_states(
		Stream& in, socket_type& connection,
		server_state& state, uint64_t client_id,
		simulator<SimulatorData>& sim)
{
	bool contains;
	client_state* cstate = state.client_states.get(client_id, contains);
	if (!contains) {
		state.client_states_lock.unlock();
		return true; /* the client was already destroyed */
	}
	cstate->lock.lock();
	state.client_states_lock.unlock();

	status response;
	uint64_t* agent_ids = nullptr;
	agent_state** agent_states = nullptr;
	size_t agent_state_count;
	bool success = true;
	if (!read(agent_state_count, in)) {
		response = status::SERVER_PARSE_MESSAGE_ERROR;
		success = false;
	} else {
		agent_ids = (uint64_t*) malloc(max((size_t) 1, sizeof(uint64_t) * agent_state_count));
		agent_states = (agent_state**) malloc(max((size_t) 1, sizeof(agent_state*) * agent_state_count));
		if (agent_ids == nullptr || agent_states == nullptr) {
			if (agent_ids != nullptr) free(agent_ids);
			response = status::SERVER_OUT_OF_MEMORY;
			success = false;
		} else if (!read(agent_ids, in, agent_state_count)) {
			free(agent_ids);
			free(agent_states);
			response = status::SERVER_PARSE_MESSAGE_ERROR;
			success = false;
		} else if (!cstate->perms.get_agent_states) {
			/* the client has no permission for this operation */
			response = status::PERMISSION_ERROR;
		} else {
			/* We have to unlock this to avoid deadlock since other simulator
				functions (i.e. `move`, `turn`, `do_nothing`) can cause the
				simulator to step. This calls `send_step_response` which needs the
				client_state locks. */
			cstate->lock.unlock();
			cstate = nullptr;

			/* make sure the agent IDs are valid */
			response = status::OK;
			for (size_t i = 0; i < agent_state_count; i++) {
				if (agent_ids[i] == 0) {
					free(agent_ids);
					free(agent_states);
					response = status::INVALID_AGENT_ID;
					break;
				}
			}

			if (response == status::OK)
				sim.get_agent_states(agent_states, agent_ids, (unsigned int) agent_state_count);
		}
	}

	memory_stream mem_stream = memory_stream(sizeof(message_type) + sizeof(response) + sizeof(size_t) + (sizeof(uint64_t) + sizeof(agent_state)) * agent_state_count);
	fixed_width_stream<memory_stream> out(mem_stream);
	success &= write(message_type::GET_AGENT_STATES_RESPONSE, out) && write(response, out)
			&& (response != status::OK || send_agent_states(out, agent_ids, agent_states, agent_state_count, sim.get_config()));
	if (response == status::OK) {
		free(agent_ids);
		free(agent_states);
	}
	if (!success) {
		if (cstate != nullptr)
			cstate->lock.unlock();
		return false;
	}

	if (cstate == nullptr) {
		cstate = acquire_client_lock(state, client_id);
		if (cstate == nullptr)
			/* the client was destroyed while we didn't have the client lock */
			return true;
	}
	success = send_message(connection, mem_stream.buffer, mem_stream.position);
	cstate->lock.unlock();
	return success;
}

/* Precondition: `state.client_states_lock` must be held by the calling thread. */
template<typename Stream, typename SimulatorData>
inline bool receive_set_active(
		Stream& in, socket_type& connection,
		server_state& state, uint64_t client_id,
		simulator<SimulatorData>& sim)
{
	bool contains;
	client_state* cstate = state.client_states.get(client_id, contains);
	if (!contains) {
		state.client_states_lock.unlock();
		return true; /* the client was already destroyed */
	}
	cstate->lock.lock();
	state.client_states_lock.unlock();

	uint64_t agent_id = UINT64_MAX;
	bool active = false, success = true;
	status response;
	if (!read(agent_id, in) || !read(active, in)) {
		response = status::SERVER_PARSE_MESSAGE_ERROR;
		success = false;
	} else if (!cstate->perms.set_active) {
		/* the client has no permission for this operation */
		response = status::PERMISSION_ERROR;
	} else {
		if (agent_id == 0 || !cstate->agent_ids.contains(agent_id)) {
			response = status::INVALID_AGENT_ID;
		} else {
			/* We have to unlock this to avoid deadlock since other simulator
			   functions (i.e. `move`, `turn`, `do_nothing`) can cause the
			   simulator to step. This calls `send_step_response` which needs the
			   client_state locks. */
			cstate->lock.unlock();
			cstate = nullptr;

			response = sim.set_agent_active(agent_id, active);
		}
	}

	memory_stream mem_stream = memory_stream(sizeof(message_type) + sizeof(agent_id) + sizeof(response));
	fixed_width_stream<memory_stream> out(mem_stream);
	success &= write(message_type::SET_ACTIVE_RESPONSE, out)
			&& write(agent_id, out) && write(response, out);
	if (!success) {
		if (cstate != nullptr)
			cstate->lock.unlock();
		return false;
	}

	if (cstate == nullptr) {
		cstate = acquire_client_lock(state, client_id);
		if (cstate == nullptr)
			/* the client was destroyed while we didn't have the client lock */
			return true;
	}
	success = send_message(connection, mem_stream.buffer, mem_stream.position);
	cstate->lock.unlock();
	return success;
}

/* Precondition: `state.client_states_lock` must be held by the calling thread. */
template<typename Stream, typename SimulatorData>
inline bool receive_is_active(
		Stream& in, socket_type& connection,
		server_state& state, uint64_t client_id,
		simulator<SimulatorData>& sim)
{
	bool contains;
	client_state* cstate = state.client_states.get(client_id, contains);
	if (!contains) {
		state.client_states_lock.unlock();
		return true; /* the client was already destroyed */
	}
	cstate->lock.lock();
	state.client_states_lock.unlock();

	bool active = false;
	uint64_t agent_id = UINT64_MAX;
	bool success = true;
	status response;
	if (!read(agent_id, in)) {
		response = status::SERVER_PARSE_MESSAGE_ERROR;
		success = false;
	} else {
		if (agent_id == 0 || !cstate->agent_ids.contains(agent_id)) {
			response = status::INVALID_AGENT_ID;
		} else {
			/* We have to unlock this to avoid deadlock since other simulator
			   functions (i.e. `move`, `turn`, `do_nothing`) can cause the
			   simulator to step. This calls `send_step_response` which needs the
			   client_state locks. */
			cstate->lock.unlock();
			cstate = nullptr;

			response = sim.is_agent_active(agent_id, active);
		}
	}

	memory_stream mem_stream = memory_stream(sizeof(message_type) + sizeof(agent_id) + sizeof(response) + sizeof(active));
	fixed_width_stream<memory_stream> out(mem_stream);
	success &= write(message_type::IS_ACTIVE_RESPONSE, out)
			&& write(agent_id, out) && write(response, out)
			&& (response != status::OK || write(active, out));
	if (!success) {
		if (cstate != nullptr)
			cstate->lock.unlock();
		return false;
	}

	if (cstate == nullptr) {
		cstate = acquire_client_lock(state, client_id);
		if (cstate == nullptr)
			/* the client was destroyed while we didn't have the client lock */
			return true;
	}
	success = send_message(connection, mem_stream.buffer, mem_stream.position);
	cstate->lock.unlock();
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
	state.client_states_lock.lock();
	if (!read(type, in)) {
		state.client_states_lock.unlock();
		return;
	}
	switch (type) {
		case message_type::ADD_AGENT:
			receive_add_agent(in, connection, state, client_id, sim); return;
		case message_type::REMOVE_AGENT:
			receive_remove_agent(in, connection, state, client_id, sim); return;
		case message_type::REMOVE_CLIENT:
			receive_remove_client(in, connection, state, client_id, sim); return;
		case message_type::ADD_SEMAPHORE:
			receive_add_semaphore(in, connection, state, client_id, sim); return;
		case message_type::REMOVE_SEMAPHORE:
			receive_remove_semaphore(in, connection, state, client_id, sim); return;
		case message_type::SIGNAL_SEMAPHORE:
			receive_signal_semaphore(in, connection, state, client_id, sim); return;
		case message_type::GET_SEMAPHORES:
			receive_get_semaphores(in, connection, state, client_id, sim); return;
		case message_type::MOVE:
			receive_move(in, connection, state, client_id, sim); return;
		case message_type::TURN:
			receive_turn(in, connection, state, client_id, sim); return;
		case message_type::DO_NOTHING:
			receive_do_nothing(in, connection, state, client_id, sim); return;
		case message_type::GET_MAP:
			receive_get_map(in, connection, state, client_id, sim); return;
		case message_type::GET_AGENT_IDS:
			receive_get_agent_ids(in, connection, state, client_id, sim); return;
		case message_type::GET_AGENT_STATES:
			receive_get_agent_states(in, connection, state, client_id, sim); return;
		case message_type::SET_ACTIVE:
			receive_set_active(in, connection, state, client_id, sim); return;
		case message_type::IS_ACTIVE:
			receive_is_active(in, connection, state, client_id, sim); return;

		case message_type::ADD_AGENT_RESPONSE:
		case message_type::REMOVE_AGENT_RESPONSE:
		case message_type::ADD_SEMAPHORE_RESPONSE:
		case message_type::REMOVE_SEMAPHORE_RESPONSE:
		case message_type::SIGNAL_SEMAPHORE_RESPONSE:
		case message_type::GET_SEMAPHORES_RESPONSE:
		case message_type::MOVE_RESPONSE:
		case message_type::TURN_RESPONSE:
		case message_type::DO_NOTHING_RESPONSE:
		case message_type::GET_MAP_RESPONSE:
		case message_type::GET_AGENT_IDS_RESPONSE:
		case message_type::GET_AGENT_STATES_RESPONSE:
		case message_type::SET_ACTIVE_RESPONSE:
		case message_type::IS_ACTIVE_RESPONSE:
		case message_type::STEP_RESPONSE:
			break;
	}
	state.client_states_lock.unlock();
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
		memory_stream mem_stream = memory_stream(sizeof(status));
		fixed_width_stream<memory_stream> out(mem_stream);
		write(status::SERVER_PARSE_MESSAGE_ERROR, out);
		send_message(connection, mem_stream.buffer, mem_stream.position);
		return false;
	}

	if (client_id == NEW_CLIENT_REQUEST) {
		std::unique_lock<std::mutex> lock(state.client_states_lock);
		if (!state.client_states.check_size()) {
			memory_stream mem_stream = memory_stream(sizeof(status));
			fixed_width_stream<memory_stream> out(mem_stream);
			write(status::SERVER_OUT_OF_MEMORY, out);
			send_message(connection, mem_stream.buffer, mem_stream.position);
			return false;
		}

		bool contains; unsigned int bucket;
		new_client.id = state.client_id_counter++;
		client_state*& cstate = state.client_states.get(new_client.id, contains, bucket);
#if !defined(NDEBUG)
		if (contains)
			fprintf(stderr, "process_new_connection WARNING: `new_client.id` already exists in `state.agent_ids`.\n");
#endif
		cstate = (client_state*) malloc(sizeof(client_state));
		if (cstate == nullptr || !init(*cstate, state.default_client_permissions)) {
			if (cstate != nullptr) free(cstate);
			memory_stream mem_stream = memory_stream(sizeof(status));
			fixed_width_stream<memory_stream> out(mem_stream);
			write(status::SERVER_OUT_OF_MEMORY, out);
			send_message(connection, mem_stream.buffer, mem_stream.position);
			return false;
		}
		state.client_states.table.keys[bucket] = new_client.id;
		state.client_states.table.size++;

		/* respond to the client */
		memory_stream mem_stream = memory_stream(sizeof(status) + sizeof(uint64_t) + sizeof(sim.time) + sizeof(simulator_config));
		fixed_width_stream<memory_stream> out(mem_stream);
		const simulator_config& config = sim.get_config();
		return write(status::OK, out)
			&& write(sim.time, out) && write(config, out)
			&& write(new_client.id, out)
			&& send_message(connection, mem_stream.buffer, mem_stream.position);

	} else {
		/* first check if the requested client ID exists */
		bool contains;
		state.client_states_lock.lock();
		client_state* cstate_ptr = state.client_states.get(client_id, contains);
		if (!contains) {
			state.client_states_lock.unlock();
			memory_stream mem_stream = memory_stream(sizeof(status));
			fixed_width_stream<memory_stream> out(mem_stream);
			write(status::INVALID_AGENT_ID, out);
			send_message(connection, mem_stream.buffer, mem_stream.position);
			return false;
		}
		client_state& cstate = *cstate_ptr;
		cstate.lock.lock();
		state.client_states_lock.unlock();
		new_client.id = client_id;

		/* respond to the client */
		memory_stream mem_stream = memory_stream(sizeof(status) + sizeof(unsigned int) + sizeof(sim.time) + sizeof(simulator_config));
		fixed_width_stream<memory_stream> out(mem_stream);
		const simulator_config& config = sim.get_config();
		if (!write(status::OK, out)
		 || !write(sim.time, out) || !write(config, out)
		 || !write(cstate.semaphore_ids.length, out)
		 || !write(cstate.semaphore_ids.data, out, cstate.semaphore_ids.length))
		{
			cstate.lock.unlock();
			fprintf(stderr, "process_new_connection ERROR: Error sending simulation time and configuration.\n");
			return false;
		}

		agent_state** agent_states = (agent_state**) malloc(max((size_t) 1, sizeof(agent_state*) * cstate.agent_ids.length));
		sim.get_agent_states(agent_states, cstate.agent_ids.data, (unsigned int) cstate.agent_ids.length);

		if (!send_agent_states(out, cstate.agent_ids.data, agent_states, cstate.agent_ids.length, config)) {
			cstate.lock.unlock();
			fprintf(stderr, "process_new_connection ERROR: Error sending agent states.\n");
			free(agent_states); return false;
		}
		free(agent_states);
		cstate.lock.unlock();

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
		const hash_map<uint64_t, agent_state*>& agents,
		const simulator_config& config,
		ExtraData&&... extra_data)
{
	std::unique_lock<std::mutex> lock(server.connection_set_lock);
	bool success = true;
	for (const auto& client_connection : server.client_connections) {
		bool contains;
		server.state.client_states_lock.lock();
		client_state* cstate = server.state.client_states.get(client_connection.value.id, contains);
		if (!contains) {
			server.state.client_states_lock.unlock();
			continue; /* this client was already destroyed */
		}
		cstate->lock.lock();
		server.state.client_states_lock.unlock();
		const array<uint64_t>& agent_ids = cstate->agent_ids;
		memory_stream mem_stream = memory_stream(sizeof(message_type) + sizeof(unsigned int) +
				(unsigned int) agent_ids.length * (sizeof(uint64_t) + sizeof(agent_state)));
		fixed_width_stream<memory_stream> out(mem_stream);
		if (!write(message_type::STEP_RESPONSE, out)) {
			cstate->lock.unlock();
			success = false;
			continue;
		}

		array<pair<uint64_t, const agent_state*>> agent_states(max((size_t) 1, agent_ids.length));
		for (uint64_t agent_id : agent_ids) {
			bool contains;
			const agent_state* agent_ptr = agents.get(agent_id, contains);
			/* `remove_agent` could have been called and `agent_ids` has not yet been updated */
			if (!contains) continue;
			agent_states[agent_states.length++] = {agent_id, agent_ptr};
		}

		bool client_success = true;
		if (!write(agent_states.length, out)) {
			client_success = false;
		} else {
			for (const auto& entry : agent_states) {
				if (!write(entry.key, out)
				 || !write(*entry.value, out, config))
				{
					client_success = false;
					break;
				}
			}
		}
		cstate->lock.unlock();
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
 * \param default_client_permissions The permissions of new clients.
 * \returns `true` if successful; `false` otherwise.
 */
template<typename SimulatorData>
bool init_server(async_server& new_server, simulator<SimulatorData>& sim,
		uint16_t server_port, unsigned int connection_queue_capacity,
		unsigned int worker_count, const permissions& default_client_permissions)
{
	std::condition_variable cv; std::mutex lock;
	auto dispatch = [&]() {
		run_server(new_server.server_socket, server_port, connection_queue_capacity,
				worker_count, new_server.status, cv, lock, new_server.client_connections,
				new_server.connection_set_lock, server_process_message<SimulatorData>,
				process_new_connection<SimulatorData>, sim, new_server.state);
	};
	new_server.status = server_status::STARTING;
	new_server.state.default_client_permissions = default_client_permissions;
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
 * \param default_client_permissions The permissions of new clients.
 * \returns `true` if successful; `false` otherwise.
 */
template<typename SimulatorData>
inline bool init_server(sync_server& new_server, simulator<SimulatorData>& sim,
		uint16_t server_port, unsigned int connection_queue_capacity,
		unsigned int worker_count, uint64_t default_client_permissions)
{
	socket_type server_socket;
	server_status dummy = server_status::STARTING;
	new_server.state.default_client_permissions = default_client_permissions;
	std::condition_variable cv; std::mutex lock;
	return run_server(
			server_socket, server_port, connection_queue_capacity,
			worker_count, dummy, cv, lock, new_server.client_connections,
			new_server.connection_set_lock, server_process_message<SimulatorData>,
			process_new_connection<SimulatorData>, sim, new_server.state);
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
 * `on_add_agent(ClientType&, uint64_t, status, agent_state&)` will be invoked,
 * where the first argument is `c`, the second is the ID of the new agent, and
 * the third is the response (OK if successful, and a different value if an
 * error occurred), and the fourth is the state of the new agent. Note the
 * fourth argument is uninitialized if an error occurred.
 *
 * \returns `true` if the sending is successful; `false` otherwise.
 */
template<typename ClientType>
bool send_add_agent(ClientType& c) {
	message_type message = message_type::ADD_AGENT;
	return send_message(c.connection, &message, sizeof(message));
}

/**
 * Sends a `remove_agent` message to the server from the client `c`. Once the
 * server responds, the function
 * `on_remove_agent(ClientType&, uint64_t, status)` will be invoked, where the
 * first argument is `c`, the second is the ID of the agent, and the third is
 * the response (OK if successful, and a different value if an error occurred).
 *
 * \returns `true` if the sending is successful; `false` otherwise.
 */
template<typename ClientType>
bool send_remove_agent(ClientType& c, uint64_t agent_id) {
	memory_stream mem_stream = memory_stream(sizeof(message_type) + sizeof(agent_id));
	fixed_width_stream<memory_stream> out(mem_stream);
	return write(message_type::REMOVE_AGENT, out)
		&& write(agent_id, out)
		&& send_message(c.connection, mem_stream.buffer, mem_stream.position);
}

/**
 * Sends an `add_semaphore` message to the server from the client `c`. Once the
 * server responds, the function
 * `on_add_semaphore(ClientType&, uint64_t, status)` will be invoked, where the
 * first argument is `c`, the second is the ID of the new semaphore, and the
 * third is the response (OK if successful, and a different value if an error
 * occurred).
 *
 * \returns `true` if the sending is successful; `false` otherwise.
 */
template<typename ClientType>
bool send_add_semaphore(ClientType& c) {
	message_type message = message_type::ADD_SEMAPHORE;
	return send_message(c.connection, &message, sizeof(message));
}

/**
 * Sends a `remove_semaphore` message to the server from the client `c`. Once
 * the server responds, the function
 * `on_remove_semaphore(ClientType&, uint64_t, status)` will be invoked, where
 * the first argument is `c`, the second is the ID of the semaphore, and the
 * third is the response (OK if successful, and a different value if an error
 * occurred).
 *
 * \returns `true` if the sending is successful; `false` otherwise.
 */
template<typename ClientType>
bool send_remove_semaphore(ClientType& c, uint64_t semaphore_id) {
	memory_stream mem_stream = memory_stream(sizeof(message_type) + sizeof(semaphore_id));
	fixed_width_stream<memory_stream> out(mem_stream);
	return write(message_type::REMOVE_SEMAPHORE, out)
		&& write(semaphore_id, out)
		&& send_message(c.connection, mem_stream.buffer, mem_stream.position);
}

/**
 * Sends a `signal_semaphore` message to the server from the client `c`. Once
 * the server responds, the function
 * `on_signal_semaphore(ClientType&, uint64_t, status)` will be invoked, where
 * the first argument is `c`, the second is `semaphore_id`, and the third is
 * the response: OK if successful, and a different value if and error occurred.
 *
 * \returns `true` if the sending is successful; `false` otherwise.
 */
template<typename ClientType>
bool send_signal_semaphore(ClientType& c, uint64_t semaphore_id) {
	memory_stream mem_stream = memory_stream(sizeof(message_type) + sizeof(semaphore_id));
	fixed_width_stream<memory_stream> out(mem_stream);
	return write(message_type::SIGNAL_SEMAPHORE, out)
		&& write(semaphore_id, out)
		&& send_message(c.connection, mem_stream.buffer, mem_stream.position);
}

/**
 * Sends a `get_semaphores` message to the server from the client `c`. Once the
 * server responds, the function
 * `on_get_semaphores(ClientType&, status, uint64_t*, bool*, size_t*)` will be
 * invoked, where the first argument is `c`, the second is the response (OK if
 * successful, and a different value if and error occurred), the third is the
 * array of semaphore IDs, and the fourth is the array of whether each
 * semaphore is signaled, and the fifth is the number of semaphores.
 *
 * \returns `true` if the sending is successful; `false` otherwise.
 */
template<typename ClientType>
bool send_get_semaphores(ClientType& c) {
	memory_stream mem_stream = memory_stream(sizeof(message_type));
	fixed_width_stream<memory_stream> out(mem_stream);
	return write(message_type::GET_SEMAPHORES, out)
		&& send_message(c.connection, mem_stream.buffer, mem_stream.position);
}

/**
 * Sends a `move` message to the server from the client `c`. Once the server
 * responds, the function `on_move(ClientType&, uint64_t, status)` will be
 * invoked, where the first argument is `c`, the second is `agent_id`, and the
 * third is the response: OK if successful, and a different value if and error
 * occurred.
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
 * responds, the function `on_turn(ClientType&, uint64_t, status)` will be
 * invoked, where the first argument is `c`, the second is `agent_id`, and the
 * third is the response: OK if successful, and a different value if an error
 * occurred.
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
 * `on_do_nothing(ClientType&, uint64_t, status)` will be invoked, where the
 * first argument is `c`, the second is `agent_id`, and the third is the
 * response: OK if successful, and a different value if an error occurred.
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
 * Sends a `get_map` message to the server from the client `c`. Once the server
 * responds, the function
 * `on_get_map(ClientType&, status, hash_map<position, patch_state>*)` will be
 * invoked, where the first argument is `c`, and the second is the response (OK
 * if successful, and a different value if an error occurred), and the third is
 * a pointer to a map containing the state information of the retrieved
 * patches. The third argument is uninitialized if the status is not OK. Memory
 * ownership of the hash_map is passed to `on_get_map`.
 *
 * \param bottom_left The bottom-left corner of the bounding box containing the
 * 		patches we wish to retrieve.
 * \param top_right The top-right corner of the bounding box containing the
 * 		patches we wish to retrieve.
 * \param get_scent_map Whether we want to also retrieve the array of scent
 * 		values at every cell in each patch. If `false`, the `scent` field will
 * 		be set to `nullptr`.
 * \param get_vision_map Whether we want to also retrieve the array of vision
 * 		values at every cell in each patch. If `false`, the `vision` field will
 * 		be set to `nullptr`.
 * \returns `true` if the sending is successful; `false` otherwise.
 */
template<typename ClientType>
bool send_get_map(ClientType& c, position bottom_left, position top_right, bool get_scent_map, bool get_vision_map) {
	memory_stream mem_stream = memory_stream(sizeof(message_type) + 2 * sizeof(position) + sizeof(get_scent_map) + sizeof(get_vision_map));
	fixed_width_stream<memory_stream> out(mem_stream);
	return write(message_type::GET_MAP, out)
		&& write(bottom_left, out) && write(top_right, out)
		&& write(get_scent_map, out) && write(get_vision_map, out)
		&& send_message(c.connection, mem_stream.buffer, mem_stream.position);
}

/**
 * Sends an `get_agent_ids` message to the server from the client `c`. Once the
 * server responds, the function
 * `on_get_agent_ids(ClientType&, status, uint64_t*, size_t)` will be
 * invoked, where the first argument is `c`, and the second is the response (OK
 * if successful, and a different value if an error occurred), the third is the
 * number of agent IDs, and the fourth is the array of agent IDs.
 *
 * \returns `true` if the sending is successful; `false` otherwise.
 */
template<typename ClientType>
bool send_get_agent_ids(ClientType& c) {
	memory_stream mem_stream = memory_stream(sizeof(message_type));
	fixed_width_stream<memory_stream> out(mem_stream);
	return write(message_type::GET_AGENT_IDS, out)
		&& send_message(c.connection, mem_stream.buffer, mem_stream.position);
}

/**
 * Sends an `get_agent_states` message to the server from the client `c`. Once
 * the server responds, the function
 * `on_get_agent_states(ClientType&, status, uint64_t*, agent_state*, size_t)`
 * will be invoked, where the first argument is `c`, and the second is the
 * response (OK if successful, and a different value if an error occurred), the
 * third is the array of agent IDs (which could be missing items from
 * `agent_ids` if the corresponding agent does not exist in the simulation),
 * the fourth is the array of agent states, and the fifth is the size of the
 * returned arrays of agent IDs and agent states.
 *
 * \param agent_ids The list of agent IDs whose states to query.
 * \param agent_id_count The number of IDs in `agent_ids`.
 * \returns `true` if the sending is successful; `false` otherwise.
 */
template<typename ClientType>
bool send_get_agent_states(ClientType& c, const uint64_t* agent_ids, size_t agent_id_count) {
	memory_stream mem_stream = memory_stream(sizeof(message_type));
	fixed_width_stream<memory_stream> out(mem_stream);
	return write(message_type::GET_AGENT_STATES, out)
		&& write(agent_id_count, out)
		&& write(agent_ids, out, agent_id_count)
		&& send_message(c.connection, mem_stream.buffer, mem_stream.position);
}

/**
 * Sends a `set_active` message to the server from the client `c`. Once the
 * server responds, the function
 * `on_set_active(ClientType&, uint64_t, status)` will be invoked, where the
 * first argument is `c`, the second is `agent_id`, and the third is the
 * response: OK if successful, and a different value if an error occurred.
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
 * `on_is_active(ClientType&, uint64_t, status, bool)` will be invoked, where
 * the first argument is `c`, and the second is `agent_id`, the third is the
 * server response (OK if active, FAILURE if inactive, or a different value if
 * an error occurred), and the fourth is whether or not the agent is active.
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
	status response;
	uint64_t agent_id = UINT64_MAX;
	bool success = true;
	agent_state& state = *((agent_state*) alloca(sizeof(agent_state)));
	fixed_width_stream<socket_type> in(c.connection);
	if (!read(response, in)) {
		response = status::CLIENT_PARSE_MESSAGE_ERROR;
		success = false;
	} else if (response == status::OK
			&& (!read(agent_id, in) || !read(state, in, c.config)))
	{
		response = status::CLIENT_PARSE_MESSAGE_ERROR;
		success = false;
	}
	on_add_agent(c, agent_id, response, state);
	if (response == status::OK) free(state);
	return success;
}

template<typename ClientType>
inline bool receive_remove_agent_response(ClientType& c) {
	status response;
	uint64_t agent_id = 0;
	bool success = true;
	fixed_width_stream<socket_type> in(c.connection);
	if (!read(agent_id, in) || !read(response, in)) {
		response = status::CLIENT_PARSE_MESSAGE_ERROR;
		success = false;
	}
	on_remove_agent(c, agent_id, response);
	return success;
}

template<typename ClientType>
inline bool receive_add_semaphore_response(ClientType& c) {
	status response;
	uint64_t semaphore_id = UINT64_MAX;
	bool success = true;
	fixed_width_stream<socket_type> in(c.connection);
	if (!read(response, in)) {
		response = status::CLIENT_PARSE_MESSAGE_ERROR;
		success = false;
	} else if (response == status::OK && !read(semaphore_id, in)) {
		response = status::CLIENT_PARSE_MESSAGE_ERROR;
		success = false;
	}
	on_add_semaphore(c, semaphore_id, response);
	return success;
}

template<typename ClientType>
inline bool receive_remove_semaphore_response(ClientType& c) {
	status response;
	uint64_t semaphore_id = 0;
	bool success = true;
	fixed_width_stream<socket_type> in(c.connection);
	if (!read(semaphore_id, in) || !read(response, in)) {
		response = status::CLIENT_PARSE_MESSAGE_ERROR;
		success = false;
	}
	on_remove_semaphore(c, semaphore_id, response);
	return success;
}

template<typename ClientType>
inline bool receive_signal_semaphore_response(ClientType& c) {
	status response;
	uint64_t semaphore_id = 0;
	bool success = true;
	fixed_width_stream<socket_type> in(c.connection);
	if (!read(semaphore_id, in) || !read(response, in)) {
		response = status::CLIENT_PARSE_MESSAGE_ERROR;
		success = false;
	}
	on_signal_semaphore(c, semaphore_id, response);
	return success;
}

template<typename ClientType>
inline bool receive_get_semaphores_response(ClientType& c) {
	status response;
	bool success = true;
	size_t semaphore_count = 0;
	fixed_width_stream<socket_type> in(c.connection);
	if (!read(response, in) || !read(semaphore_count, in)) {
		response = status::CLIENT_PARSE_MESSAGE_ERROR;
		success = false;
	}
	uint64_t* semaphore_ids = (uint64_t*) malloc(max((size_t) 1, sizeof(uint64_t) * semaphore_count));
	bool* semaphore_signaled = (bool*) malloc(max((size_t) 1, sizeof(bool) * semaphore_count));
	if (semaphore_ids == nullptr || semaphore_signaled == nullptr) {
		if (semaphore_ids != nullptr) free(semaphore_ids);
		response = status::CLIENT_OUT_OF_MEMORY;
		success = false;
	} else if (!read(semaphore_ids, in, semaphore_count)
			|| !read(semaphore_signaled, in, semaphore_count))
	{
		response = status::CLIENT_PARSE_MESSAGE_ERROR;
		free(semaphore_ids); free(semaphore_signaled);
		success = false;
	}
	/* ownership of `semaphore_ids` and `semaphore_signaled` is passed to the callee */
	on_get_semaphores(c, response, semaphore_ids, semaphore_signaled, semaphore_count);
	return success;
}

template<typename ClientType>
inline bool receive_move_response(ClientType& c) {
	status response;
	uint64_t agent_id = 0;
	bool success = true;
	fixed_width_stream<socket_type> in(c.connection);
	if (!read(agent_id, in) || !read(response, in)) {
		response = status::CLIENT_PARSE_MESSAGE_ERROR;
		success = false;
	}
	on_move(c, agent_id, response);
	return success;
}

template<typename ClientType>
inline bool receive_turn_response(ClientType& c) {
	status response;
	uint64_t agent_id = 0;
	bool success = true;
	fixed_width_stream<socket_type> in(c.connection);
	if (!read(agent_id, in) || !read(response, in)) {
		response = status::CLIENT_PARSE_MESSAGE_ERROR;
		success = false;
	}
	on_turn(c, agent_id, response);
	return success;
}

template<typename ClientType>
inline bool receive_do_nothing_response(ClientType& c) {
	status response;
	uint64_t agent_id = 0;
	bool success = true;
	fixed_width_stream<socket_type> in(c.connection);
	if (!read(agent_id, in) || !read(response, in)) {
		response = status::CLIENT_PARSE_MESSAGE_ERROR;
		success = false;
	}
	on_do_nothing(c, agent_id, response);
	return success;
}

template<typename ClientType>
inline bool receive_get_map_response(ClientType& c) {
	status response;
	bool success = true;
	array<array<patch_state>>* patches = NULL;
	fixed_width_stream<socket_type> in(c.connection);
	if (!read(response, in)) {
		response = status::CLIENT_PARSE_MESSAGE_ERROR;
		success = false;
	} else if (response == status::OK) {
		patches = (array<array<patch_state>>*) malloc(sizeof(array<array<patch_state>>));
		if (patches == NULL) {
			fprintf(stderr, "receive_get_map_response ERROR: Out of memory.\n");
			response = status::CLIENT_OUT_OF_MEMORY;
			success = false;
		} else if (!read(*patches, in, c.config)) {
			response = status::CLIENT_PARSE_MESSAGE_ERROR;
			free(patches); success = false;
		}
	}
	/* ownership of `patches` is passed to the callee */
	on_get_map(c, response, patches);
	return success;
}

template<typename ClientType>
inline bool receive_get_agent_ids_response(ClientType& c) {
	status response;
	bool success = true;
	size_t agent_count = 0;
	uint64_t* agent_ids = nullptr;
	fixed_width_stream<socket_type> in(c.connection);
	if (!read(response, in) || !read(agent_count, in)) {
		response = status::CLIENT_PARSE_MESSAGE_ERROR;
		success = false;
	} else if (agent_count > 0) {
		agent_ids = (uint64_t*) malloc(sizeof(uint64_t) * agent_count);
		if (agent_ids == nullptr) {
			fprintf(stderr, "receive_get_agent_ids_response ERROR: Out of memory.\n");
			response = status::CLIENT_OUT_OF_MEMORY;
			success = false;
		} else if (!read(agent_ids, in, agent_count)) {
			response = status::CLIENT_PARSE_MESSAGE_ERROR;
			free(agent_ids); success = false;
		}
	}
	/* ownership of `agent_ids` is passed to the callee */
	on_get_agent_ids(c, response, agent_ids, agent_count);
	return success;
}

template<typename Stream>
inline status read_agent_states(
		Stream& in,
		uint64_t*& agent_ids,
		agent_state*& agent_states,
		size_t& agent_count,
		const simulator_config& config)
{
	if (!read(agent_count, in)) {
		return status::CLIENT_PARSE_MESSAGE_ERROR;
	} else {
		agent_ids = (uint64_t*) malloc(max((size_t) 1, sizeof(uint64_t) * agent_count));
		agent_states = (agent_state*) malloc(max((size_t) 1, sizeof(agent_state) * agent_count));
		if (agent_ids == nullptr || agent_states == nullptr) {
			fprintf(stderr, "read_agent_states ERROR: Out of memory.\n");
			if (agent_ids != nullptr) free(agent_ids);
			return status::CLIENT_OUT_OF_MEMORY;
		} else if (!read(agent_ids, in, agent_count)) {
			free(agent_ids); free(agent_states);
			return status::CLIENT_PARSE_MESSAGE_ERROR;
		}

		for (size_t i = 0; i < agent_count; i++) {
			if (!read(agent_states[i], in, config)) {
				for (size_t j = 0; j < i; j++) free(agent_states[j]);
				free(agent_ids); free(agent_states);
				return status::CLIENT_PARSE_MESSAGE_ERROR;
			}
		}
	}
	return status::OK;
}

template<typename ClientType>
inline bool receive_get_agent_states_response(ClientType& c) {
	status response;
	bool success = true;
	size_t agent_count = 0;
	uint64_t* agent_ids = nullptr;
	agent_state* agent_states = nullptr;
	fixed_width_stream<socket_type> in(c.connection);
	if (!read(response, in)) {
		response = status::CLIENT_PARSE_MESSAGE_ERROR;
		success = false;
	} else if (response == status::OK) {
		response = read_agent_states(in, agent_ids, agent_states, agent_count, c.config);
		if (response != status::OK)
			success = false;
	}
	/* ownership of `agent_ids` and `agent_states` is passed to the callee */
	on_get_agent_states(c, response, agent_ids, agent_states, agent_count);
	return success;
}

template<typename ClientType>
inline bool receive_set_active_response(ClientType& c) {
	status response;
	uint64_t agent_id = 0;
	bool success = true;
	fixed_width_stream<socket_type> in(c.connection);
	if (!read(agent_id, in) || !read(response, in)) {
		response = status::CLIENT_PARSE_MESSAGE_ERROR;
		success = false;
	}
	on_set_active(c, agent_id, response);
	return success;
}

template<typename ClientType>
inline bool receive_is_active_response(ClientType& c) {
	bool active = false;
	status response;
	uint64_t agent_id = 0;
	bool success = true;
	fixed_width_stream<socket_type> in(c.connection);
	if (!read(agent_id, in) || !read(response, in)) {
		response = status::CLIENT_PARSE_MESSAGE_ERROR;
		success = false;
	} else if (response == status::OK && !read(active, in)) {
		response = status::CLIENT_PARSE_MESSAGE_ERROR;
		success = false;
	}
	on_is_active(c, agent_id, response, active);
	return success;
}

template<typename ClientType>
inline bool receive_step_response(ClientType& c) {
	bool success = true;
	status response = status::OK;
	array<uint64_t>& agent_ids = *((array<uint64_t>*) alloca(sizeof(array<uint64_t>)));

	fixed_width_stream<socket_type> in(c.connection);
	agent_state* agents = nullptr;
	if (!read(agent_ids.length, in)) {
		response = status::CLIENT_PARSE_MESSAGE_ERROR;
		agent_ids.data = nullptr; success = false;
	} else {
		agent_ids.data = (uint64_t*) malloc(max((size_t) 1, sizeof(uint64_t) * agent_ids.length));
		agents = (agent_state*) malloc(max((size_t) 1, sizeof(agent_state) * agent_ids.length));
		if (agents == nullptr || agent_ids.data == nullptr) {
			fprintf(stderr, "receive_step_response ERROR: Out of memory.\n");
			if (agent_ids.data == nullptr) free(agent_ids.data);
			response = status::CLIENT_OUT_OF_MEMORY;
			free(agent_ids); success = false;
		} else {
			agent_ids.capacity = agent_ids.length;
			for (unsigned int i = 0; i < agent_ids.length; i++) {
				if (!read(agent_ids[i], in)
				 || !read(agents[i], in, c.config))
				{
					for (unsigned int j = 0; j < i; j++) free(agents[j]);
					response = status::CLIENT_PARSE_MESSAGE_ERROR;
					free(agents); free(agent_ids); agents = nullptr;
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
		while (true) {
			wait_result result = wait_for_socket(c.connection, 0, 100000);
			if (!c.client_running) {
				return; /* stop_client was called */
			} else if (result == wait_result::DATA_AVAILABLE) {
				break;
			} else if (result == wait_result::DATA_UNAVAILABLE) {
				continue;
			} else {
				on_lost_connection(c);
				return;
			}
		}

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
		case message_type::REMOVE_AGENT_RESPONSE:
			receive_remove_agent_response(c); continue;
		case message_type::ADD_SEMAPHORE_RESPONSE:
			receive_add_semaphore_response(c); continue;
		case message_type::REMOVE_SEMAPHORE_RESPONSE:
			receive_remove_semaphore_response(c); continue;
		case message_type::SIGNAL_SEMAPHORE_RESPONSE:
			receive_signal_semaphore_response(c); continue;
		case message_type::GET_SEMAPHORES_RESPONSE:
			receive_get_semaphores_response(c); continue;
		case message_type::MOVE_RESPONSE:
			receive_move_response(c); continue;
		case message_type::TURN_RESPONSE:
			receive_turn_response(c); continue;
		case message_type::DO_NOTHING_RESPONSE:
			receive_do_nothing_response(c); continue;
		case message_type::GET_MAP_RESPONSE:
			receive_get_map_response(c); continue;
		case message_type::GET_AGENT_IDS_RESPONSE:
			receive_get_agent_ids_response(c); continue;
		case message_type::GET_AGENT_STATES_RESPONSE:
			receive_get_agent_states_response(c); continue;
		case message_type::SET_ACTIVE_RESPONSE:
			receive_set_active_response(c); continue;
		case message_type::IS_ACTIVE_RESPONSE:
			receive_is_active_response(c); continue;
		case message_type::STEP_RESPONSE:
			receive_step_response(c); continue;

		case message_type::ADD_AGENT:
		case message_type::REMOVE_AGENT:
		case message_type::REMOVE_CLIENT:
		case message_type::ADD_SEMAPHORE:
		case message_type::REMOVE_SEMAPHORE:
		case message_type::SIGNAL_SEMAPHORE:
		case message_type::GET_SEMAPHORES:
		case message_type::MOVE:
		case message_type::TURN:
		case message_type::DO_NOTHING:
		case message_type::GET_MAP:
		case message_type::GET_AGENT_IDS:
		case message_type::GET_AGENT_STATES:
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
		status response;
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
		uint64_t*& agent_ids, agent_state*& agent_states, unsigned int& agent_count,
		uint64_t*& semaphore_ids, unsigned int& semaphore_count)
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
			remove_client(existing_client); return false;
		}

		/* read the simulator response */
		status response;
		fixed_width_stream<socket_type> in(connection);
		if (!read(response, in)) {
			fprintf(stderr, "reconnect_client ERROR: Error receiving response from server.\n");
			remove_client(existing_client); return false;
		}

		/* read the simulator time and configuration */
		size_t s_semaphore_count;
		simulator_config& config = *((simulator_config*) alloca(sizeof(simulator_config)));
		if (!read(simulator_time, in)
		 || !read(config, in)
		 || !read(s_semaphore_count, in))
		{
			fprintf(stderr, "reconnect_client ERROR: Error receiving simulator time and configuration.\n");
			remove_client(existing_client); return false;
		}
		swap(existing_client.config, config);
		free(config); /* free the old configuration */
		semaphore_count = (unsigned int) s_semaphore_count;

		semaphore_ids = (uint64_t*) malloc(max((size_t) 1, sizeof(uint64_t) * semaphore_count));
		if (semaphore_ids == nullptr) {
			fprintf(stderr, "reconnect_client ERROR: Insufficient memory for `semaphore_ids`.\n");
			remove_client(existing_client); return false;
		} else if (!read(semaphore_ids, in, s_semaphore_count)) {
			fprintf(stderr, "reconnect_client ERROR: Error reading semaphore IDs.\n");
			remove_client(existing_client); free(semaphore_ids); return false;
		}

		size_t s_agent_count;
		if (read_agent_states(in, agent_ids, agent_states, s_agent_count, existing_client.config) != status::OK) {
			fprintf(stderr, "reconnect_client ERROR: Error reading agent states.\n");
			remove_client(existing_client); free(semaphore_ids); return false;
		}
		agent_count = (unsigned int) s_agent_count;

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
 * \param client_id The ID assigned to this new client by the server.
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
 * \param existing_client The client with which to attempt the connection.
 * \param client_id The ID of the client.
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
		uint64_t*& agent_ids, agent_state*& agent_states, unsigned int& agent_count,
		uint64_t*& semaphore_ids, unsigned int& semaphore_count)
{
	constexpr static unsigned int BUFFER_SIZE = 8;
	char port_str[BUFFER_SIZE];
	if (snprintf(port_str, BUFFER_SIZE, "%u", server_port) > (int) BUFFER_SIZE - 1)
		return false;

	return reconnect_client(existing_client, client_id, server_address, port_str,
			agent_ids, agent_states, agent_count, semaphore_ids, semaphore_count);
}

/**
 * Disconnects the given client `c` from the server. Note, however, that this
 * function does not remove the client from the server. This client's ID and
 * its agents will persist on the server, and the client may reconnect later.
 */
template<typename ClientData>
void stop_client(client<ClientData>& c) {
	c.client_running = false;
	if (c.response_listener.joinable()) {
		try {
			c.response_listener.join();
		} catch (...) { }
	}
	shutdown(c.connection.handle, 2);
}

/**
 * Sends a `remove_client` message to the server from the client `c`. This
 * removes the given client `c` from the server, causing the server to remove
 * all agents owned by this client and to remove the client from its memory.
 * The server will then disconnect from the client.
 *
 * NOTE: This function returns `true` if and only if the message is sent to the
 * server successfully. It may take some time after this function has returned
 * before the server processes the message and disconnects the client.
 */
template<typename ClientType>
bool remove_client(ClientType& c) {
	memory_stream mem_stream = memory_stream(sizeof(message_type));
	fixed_width_stream<memory_stream> out(mem_stream);
	c.client_running = false;
	if (!write(message_type::REMOVE_CLIENT, out)
	 || !send_message(c.connection, mem_stream.buffer, mem_stream.position))
	{
		return false;
	}

	if (c.response_listener.joinable()) {
		try {
			c.response_listener.join();
		} catch (...) { }
	}
	return true;
}

} /* namespace jbw */

#endif /* JBW_MPI_H_ */
