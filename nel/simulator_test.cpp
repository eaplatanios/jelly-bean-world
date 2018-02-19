#include "simulator.h"

#include <core/timer.h>
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

inline direction next_direction(position agent_position, double theta) {
	if (theta == M_PI) {
		return direction::UP;
	} else if (theta == 3 * M_PI / 2) {
		return direction::DOWN;
	} else if ((theta >= 0 && theta < M_PI)
			|| (theta > 3 * M_PI / 2 && theta < 2 * M_PI))
	{
		double slope = tan(theta);
		if (slope > 1.0) return direction::UP;
		else if (slope < -1.0) return direction::DOWN;
		else return direction::RIGHT;
	} else {
		double slope = tan(theta);
		if (slope > 1.0) return direction::DOWN;
		else if (slope < -1.0) return direction::UP;
		else return direction::LEFT;
	}
}

constexpr unsigned int agent_count = 10;
constexpr unsigned int max_time = 100;
unsigned int sim_time = 0;
std::condition_variable conditions[agent_count];
std::mutex locks[agent_count];
std::mutex print_lock;

//#define MULTITHREADED

inline bool try_move(simulator& sim, agent_state** agents,
		unsigned int i, unsigned int agent_count)
{
	direction dir = next_direction(agents[i]->current_position, (2 * M_PI * i) / agent_count);
	if (!sim.move(*agents[i], dir, 1)) {
		print_lock.lock();
		print("ERROR: Unable to move agent ", stderr);
		print(i, stderr); print(" from ", stderr);
		print(agents[i]->current_position, stderr);
		print(" in direction ", stderr);
		print(dir, stderr); print(".\n", stderr);
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
		if (try_move(sim, agents, id, agent_count)) {
			move_count++;

			std::unique_lock<std::mutex> lck(locks[id]);
			while (agents[id]->agent_acted)
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
	conditions[id].notify_one();
#endif
}

int main(int argc, const char** argv)
{
	simulator_config config;
	config.max_steps_per_movement = 1;
	config.scent_dimension = 3;
	config.color_dimension = 3;
	config.vision_range = 10;
	config.patch_size = 32;
	config.gibbs_iterations = 10;
	config.agent_color = (float*) calloc(config.color_dimension, sizeof(float));
	config.agent_color[2] = 1.0f;
	config.collision_policy = movement_conflict_policy::FIRST_COME_FIRST_SERVED;
	config.decay_param = 0.5;
	config.diffusion_param = 0.12;
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

	config.intensity_fn_arg_count = config.item_types.length;
	config.interaction_fn_arg_count = 4 * config.item_types.length * config.item_types.length + 1;
	config.intensity_fn = intensity;
	config.interaction_fn = interaction;
	config.intensity_fn_args = (float*) malloc(sizeof(float) * config.intensity_fn_arg_count);
	config.interaction_fn_args = (float*) malloc(sizeof(float) * config.interaction_fn_arg_count);
	config.intensity_fn_args[0] = -2.0f;
	config.interaction_fn_args[0] = config.item_types.length;
	set_interaction_args(config.interaction_fn_args, config.item_types.length, 0, 0, 40.0f, 200.0f, 0.0f, -40.0f);

	simulator sim(config, on_step);

	/* add the agents */
	agent_state* agents[agent_count];
	for (unsigned int i = 0; i < agent_count; i++) {
		agents[i] = sim.add_agent();
		if (agents[i] == NULL) {
			fprintf(stderr, "ERROR: Unable to add new agent.\n");
			return EXIT_FAILURE;
		}

		/* advance time by one to avoid collision at (0,0) */
		for (unsigned int j = 0; j <= i; j++)
			try_move(sim, agents, j, agent_count);
	}

#if defined(MULTITHREADED)
	bool simulation_running = true;
	std::atomic_uint move_count(0);
	std::thread clients[agent_count];
	for (unsigned int i = 0; i < agent_count; i++)
		clients[i] = std::thread([&,i]() { run_agent(sim, agents, i, move_count, simulation_running); });

	timer stopwatch;
	unsigned long long elapsed = 0;
	while (sim_time < max_time) {
		std::this_thread::sleep_for(std::chrono::seconds(2));
		elapsed += stopwatch.milliseconds();
		fprintf(stderr, "Completed %u moves: %lf simulation steps per second.\n", move_count.load(), ((double) sim_time / elapsed) * 1000);
		stopwatch.start();
	}
	simulation_running = false;
	for (unsigned int i = 0; i < agent_count; i++)
		clients[i].join();
#else
	timer stopwatch;
	unsigned long long elapsed = 0;
	for (unsigned int t = 0; t < max_time; t++) {
		for (unsigned int j = 0; j < agent_count; j++)
			try_move(sim, agents, j, agent_count);
		if (stopwatch.milliseconds() >= 1000) {
			elapsed += stopwatch.milliseconds();
			fprintf(stderr, "Completed %u moves: %lf simulation steps per second.\n", t * agent_count, ((double) sim_time / elapsed) * 1000);
			stopwatch.start();
		}
	}
#endif
}
