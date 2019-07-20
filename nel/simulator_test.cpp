#define _USE_MATH_DEFINES
#include "simulator.h"
#include "mpi.h"

#include <core/timer.h>
#include <cmath>
#include <thread>
#include <condition_variable>
#include <signal.h>

using namespace core;
using namespace nel;

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

constexpr unsigned int agent_count = 1;
constexpr unsigned int max_time = 10000000;
constexpr movement_conflict_policy collision_policy = movement_conflict_policy::FIRST_COME_FIRST_SERVED;
constexpr movement_pattern move_pattern = movement_pattern::TURNING;
unsigned int sim_time = 0;
bool agent_direction[agent_count];
bool waiting_for_server[agent_count];
position agent_positions[agent_count];
std::condition_variable conditions[agent_count];
std::mutex locks[agent_count];
std::mutex print_lock;
FILE* out = stderr;
async_server server;

//#define MULTITHREADED
//#define USE_MPI
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
		position current_position, unsigned int i,
		bool& reverse, direction& dir, bool& is_move)
{
	unsigned int counter = sim_time + 1;
	switch (move_pattern) {
	case movement_pattern::RADIAL:
		is_move = true;
		dir = next_direction(current_position, (2 * M_PI * i) / agent_count); break;
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
		simulator<empty_data>& sim,
		unsigned int i, bool& reverse)
{
	position current_position = agent_positions[i];

	direction dir; bool is_move;
	get_next_move(current_position, i, reverse, dir, is_move);

	if (is_move && !sim.move((uint64_t) i, dir, 1)) {
		print_lock.lock();
		print("ERROR: Unable to move agent ", out);
		print(i, out); print(" from ", out);
		print(current_position, out);
		print(" in direction ", out);
		print(dir, out); print(".\n", out);
		print_lock.unlock();
		return false;
	} else if (!is_move && !sim.turn((uint64_t) i, dir)) {
		print_lock.lock();
		print("ERROR: Unable to turn agent ", out);
		print(i, out); print(" at ", out);
		print(current_position, out);
		print(" in direction ", out);
		print(dir, out); print(".\n", out);
		print_lock.unlock();
		return false;
	}
	return true;
}

void run_agent(simulator<empty_data>& sim, unsigned int id,
		std::atomic_uint& move_count, bool& simulation_running)
{
	while (simulation_running) {
		waiting_for_server[id] = true;
		if (try_move(sim, id, agent_direction[id])) {
			move_count++;

			std::unique_lock<std::mutex> lck(locks[id]);
			while (waiting_for_server[id] && simulation_running)
				conditions[id].wait(lck);
			lck.unlock();
		}
	}
}

void on_step(const simulator<empty_data>* sim,
		const array<agent_state*>& agents, uint64_t time)
{
	sim_time++;

	/* get agent states */
	for (unsigned int i = 0; i < agents.length; i++)
		agent_positions[i] = agents[i]->current_position;

#if defined(USE_MPI)
	if (!send_step_response(server, agents, sim->get_config())) {
		print_lock.lock();
		fprintf(out, "on_step ERROR: send_step_response failed.\n");
		print_lock.unlock();
	}
#elif defined(MULTITHREADED)
	for (unsigned int i = 0; i < agent_count; i++) {
		std::unique_lock<std::mutex> lck(locks[i]);
		waiting_for_server[i] = false;
		conditions[i].notify_one();
	}
#endif
}

bool add_agents(simulator<empty_data>& sim)
{
	for (unsigned int i = 0; i < agent_count; i++) {
		pair<uint64_t, agent_state*> new_agent = sim.add_agent();
		if (new_agent.value == NULL) {
			fprintf(out, "add_agents ERROR: Unable to add new agent.\n");
			return false;
		}
		agent_positions[i] = new_agent.value->current_position;
		agent_direction[i] = (i <= agent_count / 2);

		/* advance time by one to avoid collision at (0,0) */
		for (unsigned int j = 0; j <= i; j++)
			try_move(sim, j, agent_direction[j]);
	}
	return true;
}

bool test_singlethreaded(const simulator_config& config)
{
	simulator<empty_data>& sim = *((simulator<empty_data>*) alloca(sizeof(simulator<empty_data>)));
	if (!init(sim, config, empty_data())) {
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

		for (unsigned int j = 0; j < agent_count; j++)
			try_move(sim, j, agent_direction[j]);
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
	for (unsigned int i = 0; i < agent_count; i++)
		clients[i] = std::thread([&,i]() { run_agent(sim, i, move_count, simulation_running); });

	timer stopwatch;
	unsigned long long elapsed = 0;
	while (sim_time < max_time) {
		std::this_thread::sleep_for(std::chrono::seconds(1));
		elapsed += stopwatch.milliseconds();
		fprintf(out, "Completed %u moves: %lf simulation steps per second.\n", move_count.load(), ((double) sim_time / elapsed) * 1000);
		stopwatch.start();
	}
	simulation_running = false;
	for (unsigned int i = 0; i < agent_count; i++) {
		conditions[i].notify_one();
		if (!clients[i].joinable()) continue;
		try {
			clients[i].join();
		} catch (...) { }
	}
	return true;
}

struct client_data {
	unsigned int index;
	uint64_t agent_id;
	const hash_map<position, patch_state>* map;

	bool action_result, waiting_for_step;
	position pos;
};

void on_add_agent(client<client_data>& c, uint64_t agent_id,
		mpi_response response, const agent_state& state)
{
	unsigned int id = c.data.index;
	std::unique_lock<std::mutex> lck(locks[id]);
	waiting_for_server[id] = false;
	c.data.agent_id = agent_id;
	agent_positions[agent_id] = state.current_position;
	conditions[id].notify_one();
}

void on_move(client<client_data>& c, uint64_t agent_id, mpi_response response)
{
	unsigned int id = c.data.index;
	std::unique_lock<std::mutex> lck(locks[id]);
	waiting_for_server[id] = false;
	c.data.action_result = (response == mpi_response::SUCCESS);
	conditions[id].notify_one();
}

void on_turn(client<client_data>& c, uint64_t agent_id, mpi_response response) {
	unsigned int id = c.data.index;
	std::unique_lock<std::mutex> lck(locks[id]);
	waiting_for_server[id] = false;
	c.data.action_result = (response == mpi_response::SUCCESS);
	conditions[id].notify_one();
}

void on_get_map(
		client<client_data>& c, mpi_response response,
		const hash_map<position, patch_state>* map)
{
	unsigned int id = c.data.index;
	std::unique_lock<std::mutex> lck(locks[id]);
	waiting_for_server[id] = false;
	c.data.map = map;
	conditions[id].notify_one();
}

void on_set_active(client<client_data>& c, uint64_t agent_id, mpi_response response) {
	unsigned int id = c.data.index;
	std::unique_lock<std::mutex> lck(locks[id]);
	waiting_for_server[id] = false;
	conditions[id].notify_one();
}

void on_is_active(client<client_data>& c, uint64_t agent_id, mpi_response response) {
	unsigned int id = c.data.index;
	std::unique_lock<std::mutex> lck(locks[id]);
	waiting_for_server[id] = false;
	c.data.action_result = (response == mpi_response::SUCCESS);
	conditions[id].notify_one();
}

void on_step(client<client_data>& c,
		mpi_response response,
		const array<uint64_t>& agent_ids,
		const agent_state* agent_states)
{
	unsigned int id = c.data.index;
	std::unique_lock<std::mutex> lck(locks[id]);
	c.data.waiting_for_step = false;
#if !defined(NDEBUG)
	if (agent_ids.length != 1 || agent_ids[0] != id)
		fprintf(stderr, "on_step ERROR: Unexpected agent ID.\n");
#endif
	agent_positions[id] = agent_states[0].current_position;
	conditions[id].notify_one();
}

void on_lost_connection(client<client_data>& c) {
	print_lock.lock();
	fprintf(out, "Client %u lost connection to server.\n", c.data.index);
	print_lock.unlock();
	c.client_running = false;
	conditions[c.data.index].notify_one();
}

inline void wait_for_server(std::condition_variable& cv,
		std::mutex& lock, bool& waiting_for_server, bool& client_running)
{
	std::unique_lock<std::mutex> lck(lock);
	while (waiting_for_server && client_running) cv.wait(lck);
}

inline bool mpi_try_move(
		client<client_data>& c, unsigned int i, bool& reverse)
{
	direction dir; bool is_move;
	get_next_move(agent_positions[i], i, reverse, dir, is_move);

	/* send move request */
	waiting_for_server[i] = true;
	if (is_move && !send_move(c, c.data.agent_id, dir, 1)) {
		print_lock.lock();
		fprintf(out, "ERROR: Unable to send move request.\n");
		print_lock.unlock();
		return false;
	} else if (!is_move && !send_turn(c, c.data.agent_id, dir)) {
		print_lock.lock();
		fprintf(out, "ERROR: Unable to send turn request.\n");
		print_lock.unlock();
		return false;
	}
	wait_for_server(conditions[i], locks[i], waiting_for_server[i], c.client_running);
	if (!c.client_running) return true;

	if (!c.data.action_result) {
		print_lock.lock();
		if (is_move) {
			print("ERROR: Unable to move agent ", out);
			print(i, out); print(" from ", out);
			print(c.data.pos, out);
			print(" in direction ", out);
			print(dir, out); print(".\n", out);
		} else {
			print("ERROR: Unable to turn agent ", out);
			print(i, out); print(" at ", out);
			print(c.data.pos, out);
			print(" in direction ", out);
			print(dir, out); print(".\n", out);
		}
		print_lock.unlock();
		return false;
	}
	return true;
}

void run_mpi_agent(unsigned int id,
		client<client_data>* clients,
		std::atomic_uint& move_count)
{
	while (clients[id].client_running) {
		clients[id].data.waiting_for_step = true;
		if (mpi_try_move(clients[id], id, agent_direction[id])) {
			move_count++;
			wait_for_server(conditions[id], locks[id], clients[id].data.waiting_for_step, clients[id].client_running);
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
	if (!init_server(server, sim, 54353, 16, 4)) {
		fprintf(out, "ERROR: init_server returned false.\n");
		return false;
	}

	/* below is client-side code */
	client<client_data> clients[agent_count];
	for (unsigned int i = 0; i < agent_count; i++) {
		clients[i].data.index = i;
		uint64_t simulator_time = init_client(clients[i], "localhost", "54353", NULL, NULL, 0);
		if (simulator_time == UINT64_MAX) {
			fprintf(out, "ERROR: Unable to initialize client %u.\n", i);
			cleanup_mpi(clients, i); return false;
		}

		/* each client adds one agent to the simulation */
		waiting_for_server[i] = true;
		if (!send_add_agent(clients[i])) {
			fprintf(out, "ERROR: Unable to send add_agent request.\n");
			cleanup_mpi(clients, i); return false;
		}
		/* wait for response from server */
		wait_for_server(conditions[i], locks[i], waiting_for_server[i], clients[i].client_running);

		if (clients[i].data.agent_id == UINT64_MAX) {
			fprintf(out, "ERROR: Server returned failure for add_agent request.\n");
			cleanup_mpi(clients, i); return false;
		}

		/* advance time by one to avoid collision at (0,0) */
		for (unsigned int j = 0; j <= i; j++) {
			clients[j].data.waiting_for_step = true;
			if (!mpi_try_move(clients[j], j, agent_direction[j])) {
				cleanup_mpi(clients, i); return false;
			}
		}
		for (unsigned int j = 0; j <= i; j++)
			wait_for_server(conditions[j], locks[j], clients[j].data.waiting_for_step, clients[j].client_running);
	}

	std::atomic_uint move_count(0);
	std::thread client_threads[agent_count];
	for (unsigned int i = 0; i < agent_count; i++)
		client_threads[i] = std::thread([&,i]() { run_mpi_agent(i, clients, move_count); });

	timer stopwatch;
	unsigned long long elapsed = 0;
	while (server.state != server_state::STOPPING && sim_time < max_time)
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
		clients[i].client_running = false;
		conditions[i].notify_one();
		try {
			client_threads[i].join();
		} catch (...) { }
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
	config.vision_range = 10;
	for (unsigned int i = 0; i < (size_t) direction::COUNT; i++)
		config.allowed_movement_directions[i] = true;
	for (unsigned int i = 0; i < (size_t) direction::COUNT; i++)
		config.allowed_rotations[i] = true;
	config.patch_size = 32;
	config.gibbs_iterations = 10;
	config.agent_color = (float*) calloc(config.color_dimension, sizeof(float));
	config.agent_color[2] = 1.0f;
	config.collision_policy = collision_policy;
	config.decay_param = 0.5f;
	config.diffusion_param = 0.12f;
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
	config.item_types[1].name = "onion";
	config.item_types[1].scent = (float*) calloc(config.scent_dimension, sizeof(float));
	config.item_types[1].color = (float*) calloc(config.color_dimension, sizeof(float));
	config.item_types[1].required_item_counts = (unsigned int*) calloc(item_type_count, sizeof(unsigned int));
	config.item_types[1].required_item_costs = (unsigned int*) calloc(item_type_count, sizeof(unsigned int));
	config.item_types[1].scent[0] = 1.0f;
	config.item_types[1].color[0] = 1.0f;
	config.item_types[1].required_item_counts[1] = 1;
	config.item_types[1].blocks_movement = false;
	config.item_types[2].name = "jellybean";
	config.item_types[2].scent = (float*) calloc(config.scent_dimension, sizeof(float));
	config.item_types[2].color = (float*) calloc(config.color_dimension, sizeof(float));
	config.item_types[2].required_item_counts = (unsigned int*) calloc(item_type_count, sizeof(unsigned int));
	config.item_types[2].required_item_costs = (unsigned int*) calloc(item_type_count, sizeof(unsigned int));
	config.item_types[2].scent[2] = 1.0f;
	config.item_types[2].color[2] = 1.0f;
	config.item_types[2].blocks_movement = false;
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
}
