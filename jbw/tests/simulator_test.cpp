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

#define _USE_MATH_DEFINES
#include <jbw/simulator.h>
#include <jbw/mpi.h>

#include <core/timer.h>
#include <cmath>
#include <thread>
#include <condition_variable>
#include <signal.h>

using namespace core;
using namespace jbw;

inline void set_interaction_args(
		item_properties* item_types, unsigned int first_item_type,
		unsigned int second_item_type, interaction_function interaction,
		std::initializer_list<float> args)
{
	item_types[first_item_type].interaction_fns[second_item_type].fn = interaction;
	item_types[first_item_type].interaction_fns[second_item_type].arg_count = (unsigned int) args.size();
	item_types[first_item_type].interaction_fns[second_item_type].args = (float*) malloc(max((size_t) 1, sizeof(float) * args.size()));

	unsigned int counter = 0;
	for (auto i = args.begin(); i != args.end(); i++)
		item_types[first_item_type].interaction_fns[second_item_type].args[counter++] = *i;
}

enum class movement_pattern {
	RADIAL,
	BACK_AND_FORTH,
	TURNING
};

struct local_agent_state {
	bool direction_flag;
	position agent_position;
	uint64_t client_id;

	void operator = (const local_agent_state& src) {
		direction_flag = src.direction_flag;
		agent_position = src.agent_position;
		client_id = src.client_id;
		waiting_for_server = src.waiting_for_server;
		new (&lock) std::mutex();
		new (&condition) std::condition_variable();
	}

	static inline void move(const local_agent_state& src, local_agent_state& dst) {
		const char* src_data = (const char*) &src;
		char* dst_data = (char*) &dst;
		for (unsigned int i = 0; i < sizeof(local_agent_state); i++)
			dst_data[i] = src_data[i];
	}

	static inline void free(local_agent_state& state) {
		state.lock.~mutex();
		state.condition.~condition_variable();
	}

	bool waiting_for_server;
	std::mutex lock;
	std::condition_variable condition;
};

inline bool init(local_agent_state& state) {
	new (&state.lock) std::mutex();
	new (&state.condition) std::condition_variable();
	return true;
}

constexpr unsigned int agent_count = 1;
constexpr unsigned int max_time = 1000;
constexpr movement_conflict_policy collision_policy = movement_conflict_policy::FIRST_COME_FIRST_SERVED;
constexpr movement_pattern move_pattern = movement_pattern::TURNING;
unsigned int sim_time = 0;
hash_map<uint64_t, local_agent_state*> agent_states(agent_count * RESIZE_THRESHOLD_INVERSE);
std::mutex print_lock;
FILE* out = stderr;
async_server server;

//#define MULTITHREADED
#define USE_MPI
//#define TEST_SERIALIZATION
//#define TEST_SERVER_CONNECTION_LOSS
//#define TEST_CLIENT_CONNECTION_LOSS

inline direction next_direction(position agent_position, double theta) {
	if (theta == M_PI) {
		return direction::UP;
	} else if (theta == 3 * M_PI / 2) {
		return direction::DOWN;
	} else if ((theta >= 0 && theta < M_PI)
			|| (theta > 3 * M_PI / 2 && theta < 2 * M_PI))
	{
		double slope = tan(theta);
		if (slope * (agent_position.x + 0.5) > agent_position.y + 0.5) return direction::UP;
		else if (slope * (agent_position.x + 0.5) < agent_position.y - 0.5) return direction::DOWN;
		else return direction::RIGHT;
	} else {
		double slope = tan(theta);
		if (slope * (agent_position.x - 0.5) > agent_position.y + 0.5) return direction::UP;
		else if (slope * (agent_position.x - 0.5) < agent_position.y - 0.5) return direction::DOWN;
		else return direction::LEFT;
	}
}

inline direction next_direction(position agent_position,
		int64_t min_x, int64_t max_x, bool& reverse)
{
	if (!reverse && agent_position.x >= max_x) {
		reverse = true;
		return direction::LEFT;
	} else if (reverse && agent_position.x <= min_x) {
		reverse = false;
		return direction::RIGHT;
	} else if (!reverse) {
		return direction::RIGHT;
	} else {
		return direction::LEFT;
	}
}

inline void get_next_move(
		position current_position, uint64_t id,
		bool& reverse, direction& dir, bool& is_move)
{
	unsigned int counter = sim_time + 1;
	switch (move_pattern) {
	case movement_pattern::RADIAL:
		is_move = true;
		dir = next_direction(current_position, (2 * M_PI * (id - 1)) / agent_count); break;
	case movement_pattern::BACK_AND_FORTH:
		is_move = true;
		dir = next_direction(current_position, -10 * (int64_t) agent_count, 10 * agent_count, reverse); break;
	case movement_pattern::TURNING:
		if (counter % 20 == 0) {
			is_move = false;
			dir = direction::LEFT;
		} else if (counter % 20 == 5) {
			is_move = false;
			dir = direction::LEFT;
		} else if (counter % 20 == 10) {
			is_move = false;
			dir = direction::RIGHT;
		} else if (counter % 20 == 15) {
			is_move = false;
			dir = direction::RIGHT;
		} else {
			is_move = true;
			dir = direction::UP;
		}
	}
}

inline bool try_move(
		simulator<empty_data>& sim, uint64_t id,
		position agent_position, bool& reverse)
{
	direction dir; bool is_move;
	get_next_move(agent_position, id, reverse, dir, is_move);

	if (is_move && sim.move(id, dir, 1) != status::OK) {
		print_lock.lock();
		print("ERROR: Unable to move agent ", out);
		print(id, out); print(" from ", out);
		print(agent_position, out);
		print(" in direction ", out);
		print(dir, out); print(".\n", out);
		print_lock.unlock();
		return false;
	} else if (!is_move && sim.turn(id, dir) != status::OK) {
		print_lock.lock();
		print("ERROR: Unable to turn agent ", out);
		print(id, out); print(" at ", out);
		print(agent_position, out);
		print(" in direction ", out);
		print(dir, out); print(".\n", out);
		print_lock.unlock();
		return false;
	}
	return true;
}

void run_agent(simulator<empty_data>& sim,
	uint64_t agent_id, local_agent_state& agent,
	std::atomic_uint& move_count,
	bool& simulation_running)
{
	while (simulation_running) {
		agent.waiting_for_server = true;
		if (try_move(sim, agent_id, agent.agent_position, agent.direction_flag)) {
			move_count++;

			std::unique_lock<std::mutex> lck(agent.lock);
			while (agent.waiting_for_server && simulation_running) agent.condition.wait(lck);
		}
	}
}

void on_step(const simulator<empty_data>* sim,
		const hash_map<uint64_t, agent_state*>& agents, uint64_t time)
{
	sim_time++;

	/* get agent states */
	for (const auto& entry : agents)
		agent_states.get(entry.key)->agent_position = entry.value->current_position;

#if defined(USE_MPI)
	if (!send_step_response(server, agents, sim->get_config())) {
		print_lock.lock();
		fprintf(out, "on_step ERROR: send_step_response failed.\n");
		print_lock.unlock();
	}
#elif defined(MULTITHREADED)
	for (const auto& entry : agents) {
		local_agent_state* agent = agent_states.get(entry.key);
		std::unique_lock<std::mutex> lck(agent->lock);
		agent->waiting_for_server = false;
		agent->condition.notify_one();
	}
#endif
}

bool add_agents(simulator<empty_data>& sim)
{
	for (unsigned int i = 0; i < agent_count; i++) {
		uint64_t new_agent_id; agent_state* new_agent;
		status result = sim.add_agent(new_agent_id, new_agent);
		local_agent_state* new_agent_state = (local_agent_state*) malloc(sizeof(local_agent_state));
		if (result != status::OK || new_agent_state == nullptr || !init(*new_agent_state)) {
			fprintf(out, "add_agents ERROR: Unable to add new agent.\n");
			if (new_agent_state != nullptr) free(new_agent_state);
			return false;
		}
		new_agent_state->agent_position = new_agent->current_position;
		new_agent_state->direction_flag = (i <= agent_count / 2);
		new_agent_state->waiting_for_server = false;
		agent_states.put(new_agent_id, new_agent_state);

		/* advance time by one to avoid collision at (0,0) */
		for (const auto& entry : agent_states)
			try_move(sim, entry.key, entry.value->agent_position, entry.value->direction_flag);
	}
	return true;
}

bool test_singlethreaded(const simulator_config& config)
{
	simulator<empty_data>& sim = *((simulator<empty_data>*) alloca(sizeof(simulator<empty_data>)));
	if (init(sim, config, empty_data()) != status::OK) {
		fprintf(stderr, "ERROR: Unable to initialize simulator.\n");
		return false;
	}

	if (!add_agents(sim)) {
		free(sim); return false;
	}

	timer stopwatch;
	std::atomic_uint move_count(0);
	unsigned long long elapsed = 0;
	for (unsigned int t = 0; t < max_time; t++) {
#if defined(TEST_SERIALIZATION)
		if (t % 50 == 0) {
			char filename[1024];
			snprintf(filename, 1024, "simulator_state%u", t);
			FILE* file = open_file(filename, "wb");
			fixed_width_stream<FILE*> out(file);
			if (!write(sim, out))
				fprintf(stderr, "ERROR: write failed.\n");
			fclose(file);

			/* end the simulation and restart it by reading from file */
			free(sim);
			file = open_file(filename, "rb");
			fixed_width_stream<FILE*> in(file);
			if (!read(sim, in, empty_data())) {
				fprintf(stderr, "ERROR: read failed.\n");
				free(sim); return false;
			}
			fclose(file);
		}
#endif

		for (const auto& entry : agent_states)
			try_move(sim, entry.key, entry.value->agent_position, entry.value->direction_flag);
		move_count += agent_count;
		if (stopwatch.milliseconds() >= 1000) {
			elapsed += stopwatch.milliseconds();
			fprintf(out, "Completed %u moves: %lf simulation steps per second.\n", move_count.load(), ((double) sim_time / elapsed) * 1000);
			stopwatch.start();
		}
	}
	elapsed += stopwatch.milliseconds();
	fprintf(out, "Completed %u moves: %lf simulation steps per second.\n", move_count.load(), ((double) sim_time / elapsed) * 1000);
	free(sim);
	return true;
}

bool test_multithreaded(const simulator_config& config)
{
	simulator<empty_data> sim(config, empty_data());

	if (!add_agents(sim))
		return false;

	std::atomic_uint move_count(0);
	bool simulation_running = true;
	std::thread clients[agent_count];
	unsigned int i = 0;
	for (auto entry : agent_states) {
		clients[i] = std::thread([&,entry]() {
			run_agent(sim, entry.key, *entry.value, move_count, simulation_running);
		});
		i++;
	}

	timer stopwatch;
	unsigned long long elapsed = 0;
	while (sim_time < max_time) {
		std::this_thread::sleep_for(std::chrono::seconds(1));
		elapsed += stopwatch.milliseconds();
		fprintf(out, "Completed %u moves: %lf simulation steps per second.\n", move_count.load(), ((double) sim_time / elapsed) * 1000);
		stopwatch.start();
	}
	simulation_running = false;
	for (auto entry : agent_states)
		entry.value->condition.notify_one();
	for (unsigned int i = 0; i < agent_count; i++) {
		if (clients[i].joinable()) {
			try {
				clients[i].join();
			} catch (...) { }
		}
	}
	return true;
}

struct client_data {
	template<typename T>
	struct fixed_array {
		const T* data;
		size_t length;
	};

	struct agent_state_array {
		const uint64_t* ids;
		const agent_state* states;
		size_t length;
	};

	struct semaphore_array {
		const uint64_t* ids;
		const bool* signaled;
		size_t length;
	};

	uint64_t client_id;
	uint64_t semaphore_id;
	const array<array<patch_state>>* map;
	fixed_array<uint64_t> agent_ids;
	agent_state_array agent_states;
	semaphore_array semaphores;
	bool waiting_for_server;
	std::mutex lock;
	std::condition_variable condition;

	bool action_result, waiting_for_step;
	uint64_t agent_id;
	position pos;
};

void on_add_agent(
		client<client_data>& c, uint64_t agent_id,
		status response, const agent_state& state)
{
	std::unique_lock<std::mutex> lck(c.data.lock);
	c.data.waiting_for_server = false;
	c.data.agent_id = agent_id;
	if (response == status::OK) {
		local_agent_state* agent = (local_agent_state*) malloc(sizeof(local_agent_state));
		if (agent == nullptr || !init(*agent)) {
			fprintf(stderr, "on_add_agent ERROR: Out of memory.\n");
			if (agent == nullptr) free(agent);
		} else {
			agent->client_id = c.data.client_id;
			agent->agent_position = state.current_position;
			agent->direction_flag = ((agent_id - 1) <= agent_count / 2);
			agent_states.put(agent_id, agent);
		}
	} else {
		c.data.agent_id = UINT64_MAX;
	}
	c.data.condition.notify_one();
}

void on_remove_agent(client<client_data>& c,
		uint64_t agent_id, status response)
{
	std::unique_lock<std::mutex> lck(c.data.lock);
	c.data.waiting_for_server = false;
	bool contains; unsigned int bucket;
	local_agent_state* agent = agent_states.get(agent_id, contains, bucket);
	if (contains) {
		agent_states.remove_at(bucket);
		free(*agent); free(agent);
	}
	c.data.condition.notify_one();
}

void on_add_semaphore(client<client_data>& c,
		uint64_t semaphore_id, status response)
{
	std::unique_lock<std::mutex> lck(c.data.lock);
	c.data.waiting_for_server = false;
	c.data.semaphore_id = (response == status::OK ? semaphore_id : 0);
	c.data.condition.notify_one();
}

void on_remove_semaphore(client<client_data>& c,
		uint64_t semaphore_id, status response)
{
	std::unique_lock<std::mutex> lck(c.data.lock);
	c.data.waiting_for_server = false;
	c.data.action_result = (response == status::OK);
	c.data.condition.notify_one();
}

void on_signal_semaphore(client<client_data>& c,
		uint64_t semaphore_id, status response)
{
	std::unique_lock<std::mutex> lck(c.data.lock);
	c.data.waiting_for_server = false;
	c.data.action_result = (response == status::OK);
	c.data.condition.notify_one();
}

void on_get_semaphores(client<client_data>& c, status response,
		uint64_t* semaphore_ids, bool* signaled, size_t semaphore_count)
{
	std::unique_lock<std::mutex> lck(c.data.lock);
	c.data.waiting_for_server = false;
	c.data.semaphores.ids = semaphore_ids;
	c.data.semaphores.signaled = signaled;
	c.data.semaphores.length = semaphore_count;
	c.data.action_result = (response == status::OK);
	c.data.condition.notify_one();
}

void on_move(client<client_data>& c, uint64_t agent_id, status response)
{
	std::unique_lock<std::mutex> lck(c.data.lock);
	c.data.waiting_for_server = false;
	c.data.action_result = (response == status::OK);
	c.data.condition.notify_one();
}

void on_turn(client<client_data>& c, uint64_t agent_id, status response) {
	std::unique_lock<std::mutex> lck(c.data.lock);
	c.data.waiting_for_server = false;
	c.data.action_result = (response == status::OK);
	c.data.condition.notify_one();
}

void on_do_nothing(client<client_data>& c, uint64_t agent_id, status response) {
	std::unique_lock<std::mutex> lck(c.data.lock);
	c.data.waiting_for_server = false;
	c.data.action_result = (response == status::OK);
	c.data.condition.notify_one();
}

void on_get_map(
		client<client_data>& c, status response,
		const array<array<patch_state>>* map)
{
	std::unique_lock<std::mutex> lck(c.data.lock);
	c.data.waiting_for_server = false;
	c.data.map = map;
	c.data.condition.notify_one();
}

void on_get_agent_ids(
		client<client_data>& c, status response,
		const uint64_t* agent_ids, size_t count)
{
	std::unique_lock<std::mutex> lck(c.data.lock);
	c.data.waiting_for_server = false;
	c.data.agent_ids.data = agent_ids;
	c.data.agent_ids.length = count;
	c.data.condition.notify_one();
}

void on_get_agent_states(
		client<client_data>& c, status response,
		const uint64_t* agent_ids,
		const agent_state* agent_states, size_t count)
{
	std::unique_lock<std::mutex> lck(c.data.lock);
	c.data.waiting_for_server = false;
	c.data.agent_states.ids = agent_ids;
	c.data.agent_states.states = agent_states;
	c.data.agent_states.length = count;
	c.data.condition.notify_one();
}

void on_set_active(client<client_data>& c, uint64_t agent_id, status response) {
	std::unique_lock<std::mutex> lck(c.data.lock);
	c.data.waiting_for_server = false;
	c.data.condition.notify_one();
}

void on_is_active(client<client_data>& c, uint64_t agent_id, status response, bool active) {
	std::unique_lock<std::mutex> lck(c.data.lock);
	c.data.waiting_for_server = false;
	c.data.action_result = active;
	c.data.condition.notify_one();
}

void on_step(client<client_data>& c,
		status response,
		const array<uint64_t>& agent_ids,
		const agent_state* agent_state_array)
{
	std::unique_lock<std::mutex> lck(c.data.lock);
	c.data.waiting_for_step = false;
	for (unsigned int i = 0; i < agent_ids.length; i++) {
		local_agent_state& agent = *agent_states.get(agent_ids[i]);
		agent.agent_position = agent_state_array[i].current_position;
	}
	c.data.condition.notify_one();
}

void on_lost_connection(client<client_data>& c) {
	print_lock.lock();
	fprintf(out, "Client %" PRIu64 " lost connection to server.\n", c.data.client_id);
	print_lock.unlock();
	c.client_running = false;
	c.data.condition.notify_one();
}

inline void wait_for_server(std::condition_variable& cv,
		std::mutex& lock, bool& waiting_for_server, bool& client_running)
{
	std::unique_lock<std::mutex> lck(lock);
	while (waiting_for_server && client_running) cv.wait(lck);
}

inline bool mpi_try_move(
		client<client_data>& c, uint64_t agent_id,
		position agent_position, bool& reverse)
{
	direction dir; bool is_move;
	get_next_move(agent_position, agent_id, reverse, dir, is_move);

	/* send move request */
	c.data.waiting_for_server = true;
	if (is_move && !send_move(c, agent_id, dir, 1)) {
		print_lock.lock();
		fprintf(out, "ERROR: Unable to send move request.\n");
		print_lock.unlock();
		return false;
	} else if (!is_move && !send_turn(c, agent_id, dir)) {
		print_lock.lock();
		fprintf(out, "ERROR: Unable to send turn request.\n");
		print_lock.unlock();
		return false;
	}
	wait_for_server(c.data.condition, c.data.lock, c.data.waiting_for_server, c.client_running);
	if (!c.client_running) return true;

	if (!c.data.action_result) {
		print_lock.lock();
		if (is_move) {
			print("ERROR: Unable to move agent ", out);
			print(agent_id, out); print(" from ", out);
			print(c.data.pos, out);
			print(" in direction ", out);
			print(dir, out); print(".\n", out);
		} else {
			print("ERROR: Unable to turn agent ", out);
			print(agent_id, out); print(" at ", out);
			print(c.data.pos, out);
			print(" in direction ", out);
			print(dir, out); print(".\n", out);
		}
		print_lock.unlock();
		return false;
	}
	return true;
}

void run_mpi_agent(uint64_t agent_id,
		local_agent_state& agent,
		client<client_data>& c,
		std::atomic_uint& move_count)
{
	while (c.client_running) {
		c.data.waiting_for_step = true;
		if (mpi_try_move(c, agent_id, agent.agent_position, agent.direction_flag)) {
			move_count++;
			wait_for_server(c.data.condition, c.data.lock, c.data.waiting_for_step, c.client_running);
		}
	}
}

void cleanup_mpi(client<client_data>* clients,
		unsigned int length = agent_count)
{
	for (unsigned int i = 0; i < length; i++)
		stop_client(clients[i]);
	stop_server(server);
}

bool test_mpi(const simulator_config& config)
{
	simulator<empty_data> sim(config, empty_data());
	if (!init_server(server, sim, 54353, 16, 4, permissions::grant_all())) {
		fprintf(out, "ERROR: init_server returned false.\n");
		return false;
	}

	/* below is client-side code */
	client<client_data> clients[agent_count];
	uint64_t client_ids[agent_count];
	for (unsigned int i = 0; i < agent_count; i++) {
		uint64_t simulator_time = connect_client(clients[i], "localhost", "54353", client_ids[i]);
		if (simulator_time == UINT64_MAX) {
			fprintf(out, "ERROR: Unable to initialize client %u.\n", i);
			cleanup_mpi(clients, i); return false;
		}

		/* each client adds one agent to the simulation */
		clients[i].data.waiting_for_server = true;
		clients[i].data.client_id = client_ids[i];
		if (!send_add_agent(clients[i])) {
			fprintf(out, "ERROR: Unable to send add_agent request.\n");
			cleanup_mpi(clients, i); return false;
		}
		/* wait for response from server */
		wait_for_server(clients[i].data.condition, clients[i].data.lock, clients[i].data.waiting_for_server, clients[i].client_running);

		if (clients[i].data.agent_id == UINT64_MAX) {
			fprintf(out, "ERROR: Server returned failure for add_agent request.\n");
			cleanup_mpi(clients, i); return false;
		}

		/* advance time by one to avoid collision at (0,0) */
		for (auto entry : agent_states) {
			client<client_data>& current_client = clients[index_of(entry.value->client_id, client_ids, agent_count)];
			current_client.data.waiting_for_step = true;
			if (!mpi_try_move(current_client, entry.key, entry.value->agent_position, entry.value->direction_flag)) {
				cleanup_mpi(clients, i); return false;
			}
		}
		for (auto entry : agent_states) {
			client<client_data>& current_client = clients[index_of(entry.value->client_id, client_ids, agent_count)];
			wait_for_server(current_client.data.condition, current_client.data.lock, current_client.data.waiting_for_step, current_client.client_running);
		}
	}

	std::atomic_uint move_count(0);
	std::thread client_threads[agent_count];
	unsigned int i = 0;
	for (auto entry : agent_states) {
		unsigned int client_index = index_of(entry.value->client_id, client_ids, agent_count);
		client_threads[i] = std::thread([&,entry,client_index]() { run_mpi_agent(entry.key, *entry.value, clients[client_index], move_count); });
		i++;
	}

	timer stopwatch;
	unsigned long long elapsed = 0;
	while (server.status != server_status::STOPPING && sim_time < max_time)
	{
		if (sim_time > max_time / 2) {
#if defined(TEST_SERVER_CONNECTION_LOSS)
			/* try closing all TCP sockets */
			close(server.server_socket);
			for (socket_type& client : server.client_connections)
				close(client);
#elif defined(TEST_CLIENT_CONNECTION_LOSS)
			if (server.client_connections.size == agent_count) {
				/* try closing half of the client TCP sockets */
				unsigned int index = 0;
				for (socket_type& client : server.client_connections) {
					close(client); index++;
					if (index > agent_count / 2) break;
				}
			}
#endif
		}

		std::this_thread::sleep_for(std::chrono::seconds(1));
		elapsed += stopwatch.milliseconds();
		fprintf(out, "Completed %u moves: %lf simulation steps per second.\n", move_count.load(), ((double) sim_time / elapsed) * 1000);
		stopwatch.start();
	}
	for (unsigned int i = 0; i < agent_count; i++) {
		client<client_data>& current_client = clients[i];
		current_client.client_running = false;
		current_client.data.condition.notify_one();
	} for (unsigned int i = 0; i < agent_count; i++) {
		if (client_threads[i].joinable()) {
			try {
				client_threads[i].join();
			} catch (...) { }
		}
	}
	cleanup_mpi(clients);
	return true;
}

int main(int argc, const char** argv)
{
#if !defined(_WIN32)
	signal(SIGPIPE, SIG_IGN);
#endif
	simulator_config config;
	config.max_steps_per_movement = 1;
	config.scent_dimension = 3;
	config.color_dimension = 3;
	config.vision_range = 5;
	config.agent_field_of_view = 2.09f;
	for (unsigned int i = 0; i < (size_t) direction::COUNT; i++)
		config.allowed_movement_directions[i] = action_policy::ALLOWED;
	for (unsigned int i = 0; i < (size_t) direction::COUNT; i++)
		config.allowed_rotations[i] = action_policy::ALLOWED;
	config.no_op_allowed = false;
	config.patch_size = 32;
	config.mcmc_iterations = 4000;
	config.agent_color = (float*) calloc(config.color_dimension, sizeof(float));
	config.agent_color[2] = 1.0f;
	config.collision_policy = collision_policy;
	config.decay_param = 0.4f;
	config.diffusion_param = 0.14f;
	config.deleted_item_lifetime = 2000;

	/* configure item types */
	unsigned int item_type_count = 4;
	config.item_types.ensure_capacity(item_type_count);
	config.item_types[0].name = "banana";
	config.item_types[0].scent = (float*) calloc(config.scent_dimension, sizeof(float));
	config.item_types[0].color = (float*) calloc(config.color_dimension, sizeof(float));
	config.item_types[0].required_item_counts = (unsigned int*) calloc(item_type_count, sizeof(unsigned int));
	config.item_types[0].required_item_costs = (unsigned int*) calloc(item_type_count, sizeof(unsigned int));
	config.item_types[0].scent[1] = 1.0f;
	config.item_types[0].color[1] = 1.0f;
	config.item_types[0].required_item_counts[0] = 1;
	config.item_types[0].blocks_movement = false;
	config.item_types[0].visual_occlusion = 0.0;
	config.item_types[1].name = "onion";
	config.item_types[1].scent = (float*) calloc(config.scent_dimension, sizeof(float));
	config.item_types[1].color = (float*) calloc(config.color_dimension, sizeof(float));
	config.item_types[1].required_item_counts = (unsigned int*) calloc(item_type_count, sizeof(unsigned int));
	config.item_types[1].required_item_costs = (unsigned int*) calloc(item_type_count, sizeof(unsigned int));
	config.item_types[1].scent[0] = 1.0f;
	config.item_types[1].color[0] = 1.0f;
	config.item_types[1].required_item_counts[1] = 1;
	config.item_types[1].blocks_movement = false;
	config.item_types[1].visual_occlusion = 0.0;
	config.item_types[2].name = "jellybean";
	config.item_types[2].scent = (float*) calloc(config.scent_dimension, sizeof(float));
	config.item_types[2].color = (float*) calloc(config.color_dimension, sizeof(float));
	config.item_types[2].required_item_counts = (unsigned int*) calloc(item_type_count, sizeof(unsigned int));
	config.item_types[2].required_item_costs = (unsigned int*) calloc(item_type_count, sizeof(unsigned int));
	config.item_types[2].scent[2] = 1.0f;
	config.item_types[2].color[2] = 1.0f;
	config.item_types[2].blocks_movement = false;
	config.item_types[2].visual_occlusion = 0.0;
	config.item_types[3].name = "wall";
	config.item_types[3].scent = (float*) calloc(config.scent_dimension, sizeof(float));
	config.item_types[3].color = (float*) calloc(config.color_dimension, sizeof(float));
	config.item_types[3].required_item_counts = (unsigned int*) calloc(item_type_count, sizeof(unsigned int));
	config.item_types[3].required_item_costs = (unsigned int*) calloc(item_type_count, sizeof(unsigned int));
	config.item_types[3].color[0] = 0.5f;
	config.item_types[3].color[1] = 0.5f;
	config.item_types[3].color[2] = 0.5f;
	config.item_types[3].required_item_counts[3] = 1;
	config.item_types[3].blocks_movement = true;
	config.item_types[3].visual_occlusion = 0.5;
	config.item_types.length = item_type_count;

	config.item_types[0].intensity_fn.fn = constant_intensity_fn;
	config.item_types[0].intensity_fn.arg_count = 1;
	config.item_types[0].intensity_fn.args = (float*) malloc(sizeof(float) * 1);
	config.item_types[0].intensity_fn.args[0] = -5.3f;
	config.item_types[0].interaction_fns = (energy_function<interaction_function>*)
			malloc(sizeof(energy_function<interaction_function>) * config.item_types.length);
	config.item_types[1].intensity_fn.fn = constant_intensity_fn;
	config.item_types[1].intensity_fn.arg_count = 1;
	config.item_types[1].intensity_fn.args = (float*) malloc(sizeof(float) * 1);
	config.item_types[1].intensity_fn.args[0] = -5.0f;
	config.item_types[1].interaction_fns = (energy_function<interaction_function>*)
			malloc(sizeof(energy_function<interaction_function>) * config.item_types.length);
	config.item_types[2].intensity_fn.fn = constant_intensity_fn;
	config.item_types[2].intensity_fn.arg_count = 1;
	config.item_types[2].intensity_fn.args = (float*) malloc(sizeof(float) * 1);
	config.item_types[2].intensity_fn.args[0] = -5.3f;
	config.item_types[2].interaction_fns = (energy_function<interaction_function>*)
			malloc(sizeof(energy_function<interaction_function>) * config.item_types.length);
	config.item_types[3].intensity_fn.fn = constant_intensity_fn;
	config.item_types[3].intensity_fn.arg_count = 1;
	config.item_types[3].intensity_fn.args = (float*) malloc(sizeof(float) * 1);
	config.item_types[3].intensity_fn.args[0] = 0.0f;
	config.item_types[3].interaction_fns = (energy_function<interaction_function>*)
			malloc(sizeof(energy_function<interaction_function>) * config.item_types.length);

	set_interaction_args(config.item_types.data, 0, 0, piecewise_box_interaction_fn, {10.0f, 200.0f, 0.0f, -6.0f});
	set_interaction_args(config.item_types.data, 0, 1, piecewise_box_interaction_fn, {200.0f, 0.0f, -6.0f, -6.0f});
	set_interaction_args(config.item_types.data, 0, 2, piecewise_box_interaction_fn, {10.0f, 200.0f, 2.0f, -100.0f});
	set_interaction_args(config.item_types.data, 0, 3, zero_interaction_fn, {});
	set_interaction_args(config.item_types.data, 1, 0, piecewise_box_interaction_fn, {200.0f, 0.0f, -6.0f, -6.0f});
	set_interaction_args(config.item_types.data, 1, 1, zero_interaction_fn, {});
	set_interaction_args(config.item_types.data, 1, 2, piecewise_box_interaction_fn, {200.0f, 0.0f, -100.0f, -100.0f});
	set_interaction_args(config.item_types.data, 1, 3, zero_interaction_fn, {});
	set_interaction_args(config.item_types.data, 2, 0, piecewise_box_interaction_fn, {10.0f, 200.0f, 2.0f, -100.0f});
	set_interaction_args(config.item_types.data, 2, 1, piecewise_box_interaction_fn, {200.0f, 0.0f, -100.0f, -100.0f});
	set_interaction_args(config.item_types.data, 2, 2, piecewise_box_interaction_fn, {10.0f, 200.0f, 0.0f, -6.0f});
	set_interaction_args(config.item_types.data, 2, 3, zero_interaction_fn, {});
	set_interaction_args(config.item_types.data, 3, 0, zero_interaction_fn, {});
	set_interaction_args(config.item_types.data, 3, 1, zero_interaction_fn, {});
	set_interaction_args(config.item_types.data, 3, 2, zero_interaction_fn, {});
	set_interaction_args(config.item_types.data, 3, 3, cross_interaction_fn, {10.0f, 15.0f, 20.0f, -200.0f, -20.0f, 1.0f});

#if defined(USE_MPI)
	test_mpi(config);
#elif defined(MULTITHREADED)
	test_multithreaded(config);
#else
	test_singlethreaded(config);
#endif

	for (auto entry : agent_states) {
		free(*entry.value);
		free(entry.value);
	}
}
