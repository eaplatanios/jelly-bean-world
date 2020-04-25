#include <jbw/simulator.h>
#include <jbw/mpi.h>

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

struct server_data {
	async_server* server;
	std::mutex lock;
	std::condition_variable cv;
	bool waiting;

	static inline void free(const server_data& data) { }
};

inline bool init(server_data& data, const server_data& src) {
	data.server = src.server;
	data.waiting = false;
	new (&data.lock) std::mutex();
	new (&data.cv) std::condition_variable();
	return true;
}

inline void on_step(simulator<server_data>* sim,
		const hash_map<uint64_t, agent_state*>& agents, uint64_t time)
{
	server_data& data = sim->get_data();
    if (data.server->status != server_status::STOPPING)
		send_step_response(*data.server, agents, sim->get_config());
	data.waiting = false;
	data.cv.notify_one();
}

int main(int argc, const char** argv)
{
	set_seed(0);

	simulator_config config;
	config.max_steps_per_movement = 1;
	config.scent_dimension = 3;
	config.color_dimension = 3;
	config.vision_range = 5;
	config.agent_field_of_view = 2 * M_PI;
	config.allowed_movement_directions[0] = action_policy::ALLOWED;
	config.allowed_movement_directions[1] = action_policy::DISALLOWED;
	config.allowed_movement_directions[2] = action_policy::DISALLOWED;
	config.allowed_movement_directions[3] = action_policy::DISALLOWED;
	config.allowed_rotations[0] = action_policy::DISALLOWED;
	config.allowed_rotations[1] = action_policy::DISALLOWED;
	config.allowed_rotations[2] = action_policy::ALLOWED;
	config.allowed_rotations[3] = action_policy::ALLOWED;
	config.no_op_allowed = false;
	config.patch_size = 32;
	config.mcmc_iterations = 4000;
	config.agent_color = (float*) calloc(config.color_dimension, sizeof(float));
	config.agent_color[2] = 1.0f;
	config.collision_policy = movement_conflict_policy::FIRST_COME_FIRST_SERVED;
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
	config.item_types[3].visual_occlusion = 0.0;
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

	unsigned int jellybean_index = config.item_types.length;
	for (unsigned int i = 0; i < config.item_types.length; i++) {
		if (config.item_types[i].name == "jellybean") {
			jellybean_index = i;
			break;
		}
	}

	if (jellybean_index == config.item_types.length) {
		fprintf(stderr, "ERROR: There is no item named 'jellybean'.\n");
		return EXIT_FAILURE;
	}

	const float* jellybean_scent = config.item_types[jellybean_index].scent;

	simulator<server_data>& sim = *((simulator<server_data>*) alloca(sizeof(simulator<server_data>)));
	if (init(sim, config, server_data(), get_seed()) != status::OK) {
		fprintf(stderr, "ERROR: Unable to initialize simulator.\n");
		return EXIT_FAILURE;
	}

	async_server server;
	bool server_started = true;
	server_data& sim_data = sim.get_data();
	sim_data.server = &server;
	if (!init_server(server, sim, 54353, 256, 8, permissions::grant_all())) {
		fprintf(stderr, "WARNING: Unable to start server.\n");
		server_started = false;
	}

	uint64_t agent_id; agent_state* agent;
	sim.add_agent(agent_id, agent);

	float scent_history[2] = {0};
	position position_history[2] = {{0, 0}, {0, 0}};
	uint_fast8_t history_length = 0;
	direction move_queue[4];
	uint_fast8_t move_queue_size = 0;
	uint_fast8_t move_queue_index = 0;
	status action_result = status::OK;
	bool reversed = false;
	for (unsigned int t = 0; true; t++)
	{
		if (action_result == status::OK) {
			scent_history[1] = scent_history[0];
			scent_history[0] = 0.0f;
			for (uint_fast8_t i = 0; i < config.scent_dimension; i++)
				scent_history[0] += agent->current_scent[i] * jellybean_scent[i];
			position_history[1] = position_history[0];
			position_history[0] = agent->current_position;
			history_length = min(history_length + 1, 2);
		}

		auto dequeue_move = [&]() {
			if (move_queue[move_queue_index] == direction::UP) {
				action_result = sim.move(agent_id, direction::UP, 1);
			} else if (move_queue[move_queue_index] == direction::LEFT) {
				action_result = sim.turn(agent_id, direction::LEFT);
			} else if (move_queue[move_queue_index] == direction::RIGHT) {
				action_result = sim.turn(agent_id, direction::RIGHT);
			} else {
				fprintf(stderr, "ERROR: Invalid move in the queue.\n");
				action_result = status::INVALID_AGENT_ID;
			}
			if (action_result == status::OK) {
				++move_queue_index;
				history_length = 0;
			}
		};

		sim_data.waiting = true;
		if (move_queue_index > 0 && move_queue[move_queue_index - 1] == direction::UP && position_history[0] == position_history[1]) {
			/* our movement was blocked while we were trying to go around */
			move_queue_size = 0; move_queue_index = 0;
			move_queue[move_queue_size++] = direction::RIGHT;
			move_queue[move_queue_size++] = direction::UP;
			move_queue[move_queue_size++] = direction::LEFT;
			move_queue[move_queue_size++] = direction::UP;
			dequeue_move();
		} else if (move_queue_index < move_queue_size) {
			dequeue_move();
		} else if (history_length == 2 && position_history[0] == position_history[1]) {
			/* our movement was blocked, so try to go around */
			move_queue_size = 0; move_queue_index = 0;
			move_queue[move_queue_size++] = direction::RIGHT;
			move_queue[move_queue_size++] = direction::UP;
			move_queue[move_queue_size++] = direction::LEFT;
			dequeue_move();
		} else if (history_length == 2 && scent_history[0] <= scent_history[1] && scent_history[0] != 0.0f) {
			if (!reversed) {
				move_queue_size = 0; move_queue_index = 0;
				move_queue[move_queue_size++] = direction::RIGHT;
				move_queue[move_queue_size++] = direction::RIGHT;
				dequeue_move();
				reversed = true;
			} else {
				move_queue_size = 0; move_queue_index = 0;
				move_queue[move_queue_size++] = direction::RIGHT;
				move_queue[move_queue_size++] = direction::RIGHT;
				if (scent_history[0] < scent_history[1])
					move_queue[move_queue_size++] = direction::UP;
				move_queue[move_queue_size++] = (rand() % 2 == 0) ? direction::LEFT : direction::RIGHT;
				reversed = false;
				dequeue_move();
			}
		} else {
			action_result = sim.move(agent_id, direction::UP, 1);
		}

		if (action_result != status::OK) t--;

		std::unique_lock<std::mutex> lock(sim_data.lock);
		while (sim_data.waiting) sim_data.cv.wait(lock);
		lock.unlock();

		if (t % 1000 == 0) {
			printf("[iteration %u]\n"
				"  Agent position: ", t);
			print(agent->current_position, stdout); printf("\n  Jellybeans collected: %u\n  Reward rate: %lf\n",
					agent->collected_items[jellybean_index],
					(double) agent->collected_items[jellybean_index] / (t + 1));
			fflush(stdout);
		}
	}

	if (server_started)
		stop_server(server);
	free(sim);
}
