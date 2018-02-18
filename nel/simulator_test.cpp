#include "simulator.h"

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

void on_step(const simulator* sim, unsigned int id,
		const agent_state& agent, const simulator_config& config)
{
	print("on_step: agent position is ", stderr);
	print(agent.current_position, stderr); print('\n', stderr);
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

	agent_state* agent = sim.add_agent();
	for (unsigned int t = 0; t < 100000; t++) {
		fprintf(stderr, "time = %u\n", t);
		bool status = sim.move(*agent, direction::RIGHT, 1);
		fprintf(stderr, "move returned %s.\n", status ? "true" : "false");
	}
}
