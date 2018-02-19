
#define _USE_MATH_DEFINES
#include "simulator.h"

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
std::condition_variable conditions[agent_count];
std::mutex locks[agent_count];
std::mutex print_lock;
FILE* out = stderr;

#define MULTITHREADED

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

inline bool try_move(simulator& sim, agent_state** agents,
		unsigned int i, unsigned int agent_count, bool& reverse)
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
		if (try_move(sim, agents, id, agent_count, agent_direction[id])) {
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
#if defined(MULTITHREADED)
	std::unique_lock<std::mutex> lck(locks[id]);
	conditions[id].notify_one();
#endif
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

	/* add the agents */
	agent_state* agents[agent_count];
	for (unsigned int i = 0; i < agent_count; i++) {
		agents[i] = sim.add_agent();
		agent_direction[i] = (i <= agent_count / 2);
		if (agents[i] == NULL) {
			fprintf(out, "ERROR: Unable to add new agent.\n");
			return EXIT_FAILURE;
		}

		/* advance time by one to avoid collision at (0,0) */
		for (unsigned int j = 0; j <= i; j++)
			try_move(sim, agents, j, agent_count, agent_direction[j]);
	}

	std::atomic_uint move_count(0);
#if defined(MULTITHREADED)
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
	simulation_running = false;
	for (unsigned int i = 0; i < agent_count; i++) {
		conditions[i].notify_one();
		clients[i].join();
	}
#else
	timer stopwatch;
	unsigned long long elapsed = 0;
	for (unsigned int t = 0; t < max_time; t++) {
		for (unsigned int j = 0; j < agent_count; j++)
			try_move(sim, agents, j, agent_count, agent_direction[j]);
		move_count += 10;
		for (unsigned int j = 0; j < agent_count; j++)
			check_collisions(agents, j);
		if (stopwatch.milliseconds() >= 1000) {
			elapsed += stopwatch.milliseconds();
			fprintf(out, "Completed %u moves: %lf simulation steps per second.\n", move_count.load(), ((double) sim_time / elapsed) * 1000);
			stopwatch.start();
		}
	}
#endif
	fprintf(out, "Completed %u moves: %lf simulation steps per second.\n", move_count.load(), ((double) sim_time / elapsed) * 1000);
}
