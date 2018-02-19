
#define _USE_MATH_DEFINES
#include "simulator.h"
#include "mpi.h"

#include <core/timer.h>
#include <cmath>
#include <thread>
#include <condition_variable>

using namespace core;
using namespace nel;

float intensity(const position& world_position, unsigned int type, float* args) {
	return args[type];
}

float interaction(
		const position& first_position, const position& second_position,
		unsigned int first_type, unsigned int second_type, float* args)
{
	unsigned int item_type_count = (unsigned int) args[0];
	float first_cutoff = args[4 * (first_type * item_type_count + second_type) + 1];
	float second_cutoff = args[4 * (first_type * item_type_count + second_type) + 2];
	float first_value = args[4 * (first_type * item_type_count + second_type) + 3];
	float second_value = args[4 * (first_type * item_type_count + second_type) + 4];

	uint64_t squared_length = (first_position - second_position).squared_length();
	if (squared_length < first_cutoff)
		return first_value;
	else if (squared_length < second_cutoff)
		return second_value;
	else return 0.0;
}

inline void set_interaction_args(float* args, unsigned int item_type_count,
		unsigned int first_item_type, unsigned int second_item_type,
		float first_cutoff, float second_cutoff, float first_value, float second_value)
{
	args[4 * (first_item_type * item_type_count + second_item_type) + 1] = first_cutoff;
	args[4 * (first_item_type * item_type_count + second_item_type) + 2] = second_cutoff;
	args[4 * (first_item_type * item_type_count + second_item_type) + 3] = first_value;
	args[4 * (first_item_type * item_type_count + second_item_type) + 4] = second_value;
}

enum class movement_pattern {
	RADIAL,
	BACK_AND_FORTH
};

constexpr unsigned int agent_count = 8;
constexpr unsigned int max_time = 1000000;
constexpr movement_conflict_policy collision_policy = movement_conflict_policy::RANDOM;
constexpr movement_pattern move_pattern = movement_pattern::BACK_AND_FORTH;
unsigned int sim_time = 0;
bool agent_direction[agent_count];
bool waiting_for_server[agent_count];
std::condition_variable conditions[agent_count];
std::mutex locks[agent_count];
std::mutex print_lock;
FILE* out = stderr;
async_server server;

#define MULTITHREADED
#define USE_MPI

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

inline bool try_move(simulator& sim,
		agent_state** agents,
		unsigned int i, bool& reverse)
{
	direction dir;
	switch (move_pattern) {
	case movement_pattern::RADIAL:
		dir = next_direction(agents[i]->current_position, (2 * M_PI * i) / agent_count);
	case movement_pattern::BACK_AND_FORTH:
		dir = next_direction(agents[i]->current_position, -10 * (int64_t) agent_count, 10 * agent_count, reverse);
	}

	if (!sim.move(*agents[i], dir, 1)) {
		print_lock.lock();
		print("ERROR: Unable to move agent ", out);
		print(i, out); print(" from ", out);
		print(agents[i]->current_position, out);
		print(" in direction ", out);
		print(dir, out); print(".\n", out);
		print_lock.unlock();
		return false;
	}
	return true;
}

void run_agent(simulator& sim, agent_state** agents,
		unsigned int id, std::atomic_uint& move_count,
		bool& simulation_running)
{
	while (simulation_running) {
		if (try_move(sim, agents, id, agent_direction[id])) {
			move_count++;

			std::unique_lock<std::mutex> lck(locks[id]);
			while (agents[id]->agent_acted && simulation_running)
				conditions[id].wait(lck);
			lck.unlock();
		}
	}
}

void on_step(const simulator* sim, unsigned int id,
		const agent_state& agent, const simulator_config& config)
{
	if (id == 0) sim_time++;
#if defined(USE_MPI)
	if (!send_step_response(server, agent)) {
		print_lock.lock();
		fprintf(out, "on_step ERROR: send_step_response failed.\n");
		print_lock.unlock();
	}
#elif defined(MULTITHREADED)
	std::unique_lock<std::mutex> lck(locks[id]);
	conditions[id].notify_one();
#endif
}

bool add_agents(simulator& sim, agent_state* agents[agent_count])
{
	for (unsigned int i = 0; i < agent_count; i++) {
		agents[i] = sim.add_agent();
		if (agents[i] == NULL) {
			fprintf(out, "add_agents ERROR: Unable to add new agent.\n");
			return false;
		}
		agent_direction[i] = (i <= agent_count / 2);

		/* advance time by one to avoid collision at (0,0) */
		for (unsigned int j = 0; j <= i; j++)
			try_move(sim, agents, j, agent_direction[j]);
	}
	return true;
}

bool test_singlethreaded(simulator& sim)
{
	agent_state* agents[agent_count];
	if (!add_agents(sim, agents))
		return false;

	timer stopwatch;
	std::atomic_uint move_count(0);
	unsigned long long elapsed = 0;
	for (unsigned int t = 0; t < max_time; t++) {
		for (unsigned int j = 0; j < agent_count; j++)
			try_move(sim, agents, j, agent_direction[j]);
		move_count += 10;
		if (stopwatch.milliseconds() >= 1000) {
			elapsed += stopwatch.milliseconds();
			fprintf(out, "Completed %u moves: %lf simulation steps per second.\n", move_count.load(), ((double) sim_time / elapsed) * 1000);
			stopwatch.start();
		}
	}
	elapsed += stopwatch.milliseconds();
	fprintf(out, "Completed %u moves: %lf simulation steps per second.\n", move_count.load(), ((double) sim_time / elapsed) * 1000);
	return true;
}

bool test_multithreaded(simulator& sim) {
	agent_state* agents[agent_count];
	if (!add_agents(sim, agents))
		return false;

	std::atomic_uint move_count(0);
	bool simulation_running = true;
	std::thread clients[agent_count];
	for (unsigned int i = 0; i < agent_count; i++)
		clients[i] = std::thread([&,i]() { run_agent(sim, agents, i, move_count, simulation_running); });

	timer stopwatch;
	unsigned long long elapsed = 0;
	while (sim_time < max_time) {
		std::this_thread::sleep_for(std::chrono::seconds(1));
		elapsed += stopwatch.milliseconds();
		fprintf(out, "Completed %u moves: %lf simulation steps per second.\n", move_count.load(), ((double) sim_time / elapsed) * 1000);
		stopwatch.start();
	}
	elapsed += stopwatch.milliseconds();
	fprintf(out, "Completed %u moves: %lf simulation steps per second.\n", move_count.load(), ((double) sim_time / elapsed) * 1000);
	simulation_running = false;
	for (unsigned int i = 0; i < agent_count; i++) {
		conditions[i].notify_one();
		clients[i].join();
	}
	return true;
}

struct client_data {
	unsigned int index;
	uint64_t agent_handle;

	bool move_result, waiting_for_step;
	position pos;
};

void add_agent_callback(client<client_data>& c, uint64_t new_agent) {
	unsigned int id = c.data.index;
	std::unique_lock<std::mutex> lck(locks[id]);
	waiting_for_server[id] = false;
	c.data.agent_handle = new_agent;
	conditions[id].notify_one();
}

void move_callback(client<client_data>& c, uint64_t agent, bool request_success) {
	unsigned int id = c.data.index;
	std::unique_lock<std::mutex> lck(locks[id]);
	waiting_for_server[id] = false;
	c.data.move_result = request_success;
	conditions[id].notify_one();
}

void get_position_callback(client<client_data>& c, uint64_t agent, const position& pos) {
	unsigned int id = c.data.index;
	std::unique_lock<std::mutex> lck(locks[id]);
	waiting_for_server[id] = false;
	c.data.pos = pos;
	conditions[id].notify_one();
}

void step_done_callback(client<client_data>& c, uint64_t agent) {
	unsigned int id = c.data.index;
	std::unique_lock<std::mutex> lck(locks[id]);
	c.data.waiting_for_step = false;
	conditions[id].notify_one();
}

void lost_connection_callback(client<client_data>& c) {
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
	/* get current position */
	waiting_for_server[i] = true;
	if (!send_get_position(c, c.data.agent_handle)) {
		print_lock.lock();
		fprintf(out, "ERROR: Unable to send get_position request.\n");
		print_lock.unlock();
		return false;
	}
	wait_for_server(conditions[i], locks[i], waiting_for_server[i], c.client_running);
	if (!c.client_running) return true;

	direction dir;
	switch (move_pattern) {
	case movement_pattern::RADIAL:
		dir = next_direction(c.data.pos, (2 * M_PI * i) / agent_count);
	case movement_pattern::BACK_AND_FORTH:
		dir = next_direction(c.data.pos, -10 * (int64_t) agent_count, 10 * agent_count, reverse);
	}

	/* send move request */
	waiting_for_server[i] = true;
	if (!send_move(c, c.data.agent_handle, dir, 1)) {
		print_lock.lock();
		fprintf(out, "ERROR: Unable to send move request.\n");
		print_lock.unlock();
		return false;
	}
	wait_for_server(conditions[i], locks[i], waiting_for_server[i], c.client_running);
	if (!c.client_running) return true;

	if (!c.data.move_result) {
		print_lock.lock();
		print("ERROR: Unable to move agent ", out);
		print(i, out); print(" in direction ", out);
		print(dir, out); print(".\n", out);
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

bool test_mpi(simulator& sim)
{
	if (!init_server(server, sim, 54353, 16, 4)) {
		fprintf(out, "ERROR: init_server returned false.\n");
		return false;
	}

	/* below is client-side code */
	client<client_data> clients[agent_count];
	client_callbacks<client<client_data>> callbacks = {
			add_agent_callback, move_callback,
			get_position_callback,
			step_done_callback,
			lost_connection_callback};
	for (unsigned int i = 0; i < agent_count; i++) {
		clients[i].data.index = i;
		if (!init_client(clients[i], callbacks, "localhost", "54353")) {
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

		if (clients[i].data.agent_handle == 0) {
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
	while (sim_time < max_time) {
		std::this_thread::sleep_for(std::chrono::seconds(1));
		elapsed += stopwatch.milliseconds();
		fprintf(out, "Completed %u moves: %lf simulation steps per second.\n", move_count.load(), ((double) sim_time / elapsed) * 1000);
		stopwatch.start();
	}
	elapsed += stopwatch.milliseconds();
	fprintf(out, "Completed %u moves: %lf simulation steps per second.\n", move_count.load(), ((double) sim_time / elapsed) * 1000);
	for (unsigned int i = 0; i < agent_count; i++) {
		clients[i].client_running = false;
		conditions[i].notify_one();
		client_threads[i].join();
	}
	cleanup_mpi(clients);
	return true;
}

int main(int argc, const char** argv)
{
	//set_seed(2890104773);
	fprintf(out, "random seed: %u\n", get_seed());
	simulator_config config;
	config.max_steps_per_movement = 1;
	config.scent_dimension = 3;
	config.color_dimension = 3;
	config.vision_range = 10;
	config.patch_size = 32;
	config.gibbs_iterations = 10;
	config.agent_color = (float*) calloc(config.color_dimension, sizeof(float));
	config.agent_color[2] = 1.0f;
	config.collision_policy = collision_policy;
	config.decay_param = 0.5f;
	config.diffusion_param = 0.12f;
	config.deleted_item_lifetime = 2000;

	/* configure item types */
	config.item_types.ensure_capacity(3);
	config.item_types[0].name = "banana";
	config.item_types[0].scent = (float*) calloc(config.scent_dimension, sizeof(float));
	config.item_types[0].color = (float*) calloc(config.color_dimension, sizeof(float));
	config.item_types[0].scent[0] = 1.0f;
	config.item_types[0].color[0] = 1.0f;
	config.item_types[0].automatically_collected = true;
	config.item_types.length = 1;

	config.intensity_fn_arg_count = (unsigned int) config.item_types.length;
	config.interaction_fn_arg_count = (unsigned int) (4 * config.item_types.length * config.item_types.length + 1);
	config.intensity_fn = intensity;
	config.interaction_fn = interaction;
	config.intensity_fn_args = (float*) malloc(sizeof(float) * config.intensity_fn_arg_count);
	config.interaction_fn_args = (float*) malloc(sizeof(float) * config.interaction_fn_arg_count);
	config.intensity_fn_args[0] = -2.0f;
	config.interaction_fn_args[0] = (float) config.item_types.length;
	set_interaction_args(config.interaction_fn_args, (unsigned int) config.item_types.length, 0, 0, 40.0f, 200.0f, 0.0f, -40.0f);

	simulator sim(config, on_step);

#if defined(USE_MPI)
	test_mpi(sim);
#elif defined(MULTITHREADED)
	test_multithreaded(sim);
#else
	test_singlethreaded(sim);
#endif
}
