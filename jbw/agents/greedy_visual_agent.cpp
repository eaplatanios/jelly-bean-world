#include <jbw/simulator.h>
#include <jbw/mpi.h>
#include <set>

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

struct shortest_path_state
{
	unsigned int cost;
	int x, y;
	shortest_path_state* prev;
	direction dir;

	unsigned int reference_count;

	static inline void free(shortest_path_state& state) {
		state.reference_count--;
		if (state.reference_count == 0 && state.prev != nullptr) {
			core::free(*state.prev);
			if (state.prev->reference_count == 0)
				core::free(state.prev);
		}
	}

	struct less_than {
		inline bool operator () (const shortest_path_state* left, const shortest_path_state* right) {
			return left->cost < right->cost;
		}
	};
};

inline bool item_exists(
		const float* vision, int vision_range,
		unsigned int color_dimension,
		const float* item_color, int x, int y)
{
	unsigned int offset = ((x + vision_range) * (2*vision_range + 1) + (y + vision_range)) * color_dimension;
	float vision_length = 0.0f;
	float item_color_length = 0.0f;
	for (unsigned int i = 0; i < color_dimension; i++) {
		vision_length += vision[offset + i] * vision[offset + i];
		item_color_length += item_color[i] * item_color[i];
	}
	vision_length = sqrt(vision_length);
	item_color_length = sqrt(item_color_length);
	if (vision_length == 0.0f)
		return fabs(item_color_length) < 1.0e-5f;
	if (item_color_length == 0.0f)
		return fabs(vision_length) < 1.0e-5f;
	for (unsigned int i = 0; i < color_dimension; i++)
		if (fabs(vision[offset + i] / vision_length - item_color[i] / item_color_length) > 1.0e-5f) return false;
	return true;
}

inline void move_forward(int x, int y, direction dir, int& new_x, int& new_y) {
	new_x = x;
	new_y = y;
	if (dir == direction::UP) ++new_y;
	else if (dir == direction::DOWN) --new_y;
	else if (dir == direction::LEFT) --new_x;
	else if (dir == direction::RIGHT) ++new_x;
}

inline direction turn_left(direction dir) {
	if (dir == direction::UP) return direction::LEFT;
	else if (dir == direction::DOWN) return direction::RIGHT;
	else if (dir == direction::LEFT) return direction::DOWN;
	else if (dir == direction::RIGHT) return direction::UP;
	fprintf(stderr, "turn_left: Unrecognized direction.\n");
	exit(EXIT_FAILURE);
}

inline direction turn_right(direction dir) {
	if (dir == direction::UP) return direction::RIGHT;
	else if (dir == direction::DOWN) return direction::LEFT;
	else if (dir == direction::LEFT) return direction::UP;
	else if (dir == direction::RIGHT) return direction::DOWN;
	fprintf(stderr, "turn_right: Unrecognized direction.\n");
	exit(EXIT_FAILURE);
}

inline bool inside_fov(int x, int y, float fov) {
	if (x < 0) x = -x;
	float angle;
	if (y == 0) angle = (float) M_PI / 2;
	else if (y > 0) angle = atan((float) x / y);
	else angle = (float) M_PI + atan((float) x / y);
	return 2*angle <= fov;
}

shortest_path_state* shortest_path(
		const float* vision, int vision_range,
		const float* jellybean_color,
		const float* wall_color,
		const float* onion_color,
		unsigned int color_dimension,
		const float fov)
{
	unsigned int* smallest_costs = (unsigned int*) malloc(sizeof(unsigned int) * (2*vision_range + 1) * (2*vision_range + 1) * 4);
	for (unsigned int i = 0; i < (unsigned int) (2*vision_range + 1) * (2*vision_range + 1) * 4; i++)
		smallest_costs[i] = UINT_MAX;

	std::multiset<shortest_path_state*, shortest_path_state::less_than> queue;
	shortest_path_state* initial_state = (shortest_path_state*) malloc(sizeof(shortest_path_state));
	initial_state->cost = 0;
	initial_state->x = 0;
	initial_state->y = 0;
	initial_state->reference_count = 1;
	initial_state->prev = nullptr;
	initial_state->dir = direction::UP;
	smallest_costs[((vision_range + 0)*(2*vision_range + 1) + (vision_range + 0))*4 + (int) direction::UP] = 0;
	queue.insert(initial_state);

	shortest_path_state* shortest_path = nullptr;
	while (!queue.empty()) {
		auto first = queue.cbegin();
		shortest_path_state* state = *first;
		queue.erase(first);

		/* check if we found a jellybean */
		if (!(state->x == 0 && state->y == 0) && item_exists(vision, vision_range, color_dimension, jellybean_color, state->x, state->y)) {
			/* we found a jellybean, we can stop the search */
			shortest_path = state;
			break;
		}

		/* consider moving forward */
		int new_x, new_y;
		direction new_dir = state->dir;
		move_forward(state->x, state->y, state->dir, new_x, new_y);
		if (new_x >= -vision_range && new_x <= vision_range && new_y >= -vision_range && new_y <= vision_range) {
			/* check if there is a wall in the new position */
			if (inside_fov(new_x, new_y, fov) && !item_exists(vision, vision_range, color_dimension, wall_color, new_x, new_y) && !item_exists(vision, vision_range, color_dimension, onion_color, new_x, new_y)) {
				/* there is no wall, so continue considering this movement */
				unsigned int new_cost = state->cost + 1;
				if (new_cost < smallest_costs[((vision_range + new_x)*(2*vision_range + 1) + (vision_range + new_y))*4 + (int) new_dir]) {
					smallest_costs[((vision_range + new_x)*(2*vision_range + 1) + (vision_range + new_y))*4 + (int) new_dir] = new_cost;

					shortest_path_state* new_state = (shortest_path_state*) malloc(sizeof(shortest_path_state));
					new_state->cost = new_cost;
					new_state->x = new_x;
					new_state->y = new_y;
					new_state->dir = new_dir;
					new_state->reference_count = 1;
					new_state->prev = state;
					++state->reference_count;
					queue.insert(new_state);
				}
			}
		}

		/* consider turning left */
		unsigned int new_cost = state->cost + 1;
		new_x = state->x;
		new_y = state->y;
		new_dir = turn_left(state->dir);
		if (new_cost < smallest_costs[((vision_range + new_x)*(2*vision_range + 1) + (vision_range + new_y))*4 + (int) new_dir]) {
			smallest_costs[((vision_range + new_x)*(2*vision_range + 1) + (vision_range + new_y))*4 + (int) new_dir] = new_cost;

			shortest_path_state* new_state = (shortest_path_state*) malloc(sizeof(shortest_path_state));
			new_state->cost = new_cost;
			new_state->x = new_x;
			new_state->y = new_y;
			new_state->dir = new_dir;
			new_state->reference_count = 1;
			new_state->prev = state;
			++state->reference_count;
			queue.insert(new_state);
		}

		/* consider turning right */
		new_cost = state->cost + 1;
		new_x = state->x;
		new_y = state->y;
		new_dir = turn_right(state->dir);
		if (new_cost < smallest_costs[((vision_range + new_x)*(2*vision_range + 1) + (vision_range + new_y))*4 + (int) new_dir]) {
			smallest_costs[((vision_range + new_x)*(2*vision_range + 1) + (vision_range + new_y))*4 + (int) new_dir] = new_cost;

			shortest_path_state* new_state = (shortest_path_state*) malloc(sizeof(shortest_path_state));
			new_state->cost = new_cost;
			new_state->x = new_x;
			new_state->y = new_y;
			new_state->dir = new_dir;
			new_state->reference_count = 1;
			new_state->prev = state;
			++state->reference_count;
			queue.insert(new_state);
		}

		free(*state);
		if (state->reference_count == 0)
			free(state);
	}

	for (auto state : queue) {
		free(*state);
		if (state->reference_count == 0)
			free(state);
	}

	free(smallest_costs);
	return shortest_path;
}

int main(int argc, const char** argv)
{
	set_seed(0);

	simulator_config config;
	config.max_steps_per_movement = 1;
	config.scent_dimension = 3;
	config.color_dimension = 3;
	config.vision_range = 8;
	config.agent_field_of_view = (float) (2 * M_PI);
	config.allowed_movement_directions[0] = action_policy::ALLOWED;
	config.allowed_movement_directions[1] = action_policy::ALLOWED;
	config.allowed_movement_directions[2] = action_policy::ALLOWED;
	config.allowed_movement_directions[3] = action_policy::ALLOWED;
	config.allowed_rotations[0] = action_policy::DISALLOWED;
	config.allowed_rotations[1] = action_policy::DISALLOWED;
	config.allowed_rotations[2] = action_policy::ALLOWED;
	config.allowed_rotations[3] = action_policy::ALLOWED;
	config.no_op_allowed = false;
	config.patch_size = 64;
	config.mcmc_iterations = 10000;
	config.agent_color = (float*) calloc(config.color_dimension, sizeof(float));
	config.collision_policy = movement_conflict_policy::FIRST_COME_FIRST_SERVED;
	config.decay_param = 0.4f;
	config.diffusion_param = 0.14f;
	config.deleted_item_lifetime = 2000;

	/* configure item types */
	unsigned int item_type_count = 6;
	config.item_types.ensure_capacity(item_type_count);
	config.item_types[0].name = "banana";
	config.item_types[0].scent = (float*) calloc(config.scent_dimension, sizeof(float));
	config.item_types[0].color = (float*) calloc(config.color_dimension, sizeof(float));
	config.item_types[0].required_item_counts = (unsigned int*) calloc(item_type_count, sizeof(unsigned int));
	config.item_types[0].required_item_costs = (unsigned int*) calloc(item_type_count, sizeof(unsigned int));
	config.item_types[0].scent[0] = 1.92f;
	config.item_types[0].scent[1] = 1.76f;
	config.item_types[0].scent[2] = 0.40f;
	config.item_types[0].color[0] = 0.96f;
	config.item_types[0].color[1] = 0.88f;
	config.item_types[0].color[2] = 0.20f;
	config.item_types[0].blocks_movement = false;
	config.item_types[0].visual_occlusion = 0.0f;
	config.item_types[1].name = "onion";
	config.item_types[1].scent = (float*) calloc(config.scent_dimension, sizeof(float));
	config.item_types[1].color = (float*) calloc(config.color_dimension, sizeof(float));
	config.item_types[1].required_item_counts = (unsigned int*) calloc(item_type_count, sizeof(unsigned int));
	config.item_types[1].required_item_costs = (unsigned int*) calloc(item_type_count, sizeof(unsigned int));
	config.item_types[1].scent[0] = 0.68f;
	config.item_types[1].scent[1] = 0.01f;
	config.item_types[1].scent[2] = 0.99f;
	config.item_types[1].color[0] = 0.68f;
	config.item_types[1].color[1] = 0.01f;
	config.item_types[1].color[2] = 0.99f;
	config.item_types[1].blocks_movement = false;
	config.item_types[1].visual_occlusion = 0.0f;
	config.item_types[2].name = "jellybean";
	config.item_types[2].scent = (float*) calloc(config.scent_dimension, sizeof(float));
	config.item_types[2].color = (float*) calloc(config.color_dimension, sizeof(float));
	config.item_types[2].required_item_counts = (unsigned int*) calloc(item_type_count, sizeof(unsigned int));
	config.item_types[2].required_item_costs = (unsigned int*) calloc(item_type_count, sizeof(unsigned int));
	config.item_types[2].scent[0] = 1.64f;
	config.item_types[2].scent[1] = 0.54f;
	config.item_types[2].scent[2] = 0.40f;
	config.item_types[2].color[0] = 0.82f;
	config.item_types[2].color[1] = 0.27f;
	config.item_types[2].color[2] = 0.20f;
	config.item_types[2].blocks_movement = false;
	config.item_types[2].visual_occlusion = 0.0f;
	config.item_types[3].name = "wall";
	config.item_types[3].scent = (float*) calloc(config.scent_dimension, sizeof(float));
	config.item_types[3].color = (float*) calloc(config.color_dimension, sizeof(float));
	config.item_types[3].required_item_counts = (unsigned int*) calloc(item_type_count, sizeof(unsigned int));
	config.item_types[3].required_item_costs = (unsigned int*) calloc(item_type_count, sizeof(unsigned int));
	config.item_types[3].color[0] = 0.20f;
	config.item_types[3].color[1] = 0.47f;
	config.item_types[3].color[2] = 0.67f;
	config.item_types[3].required_item_counts[3] = 1;
	config.item_types[3].blocks_movement = true;
	config.item_types[3].visual_occlusion = 1.0f;
	config.item_types[4].name = "tree";
	config.item_types[4].scent = (float*) calloc(config.scent_dimension, sizeof(float));
	config.item_types[4].color = (float*) calloc(config.color_dimension, sizeof(float));
	config.item_types[4].required_item_counts = (unsigned int*) calloc(item_type_count, sizeof(unsigned int));
	config.item_types[4].required_item_costs = (unsigned int*) calloc(item_type_count, sizeof(unsigned int));
	config.item_types[4].scent[0] = 0.00f;
	config.item_types[4].scent[1] = 0.47f;
	config.item_types[4].scent[2] = 0.06f;
	config.item_types[4].color[0] = 0.00f;
	config.item_types[4].color[1] = 0.47f;
	config.item_types[4].color[2] = 0.06f;
	config.item_types[4].required_item_counts[4] = 1;
	config.item_types[4].blocks_movement = false;
	config.item_types[4].visual_occlusion = 0.1f;
	config.item_types[5].name = "truffle";
	config.item_types[5].scent = (float*) calloc(config.scent_dimension, sizeof(float));
	config.item_types[5].color = (float*) calloc(config.color_dimension, sizeof(float));
	config.item_types[5].required_item_counts = (unsigned int*) calloc(item_type_count, sizeof(unsigned int));
	config.item_types[5].required_item_costs = (unsigned int*) calloc(item_type_count, sizeof(unsigned int));
	config.item_types[5].scent[0] = 8.40f;
	config.item_types[5].scent[1] = 4.80f;
	config.item_types[5].scent[2] = 2.60f;
	config.item_types[5].color[0] = 0.42f;
	config.item_types[5].color[1] = 0.24f;
	config.item_types[5].color[2] = 0.13f;
	config.item_types[5].blocks_movement = false;
	config.item_types[5].visual_occlusion = 0.0f;
	config.item_types.length = item_type_count;

	config.item_types[0].intensity_fn.fn = constant_intensity_fn;
	config.item_types[0].intensity_fn.arg_count = 1;
	config.item_types[0].intensity_fn.args = (float*) malloc(sizeof(float) * 1);
	config.item_types[0].intensity_fn.args[0] = 1.5f;
	config.item_types[0].interaction_fns = (energy_function<interaction_function>*)
			malloc(sizeof(energy_function<interaction_function>) * config.item_types.length);
	config.item_types[1].intensity_fn.fn = constant_intensity_fn;
	config.item_types[1].intensity_fn.arg_count = 1;
	config.item_types[1].intensity_fn.args = (float*) malloc(sizeof(float) * 1);
	config.item_types[1].intensity_fn.args[0] = -3.0f;
	config.item_types[1].interaction_fns = (energy_function<interaction_function>*)
			malloc(sizeof(energy_function<interaction_function>) * config.item_types.length);
	config.item_types[2].intensity_fn.fn = constant_intensity_fn;
	config.item_types[2].intensity_fn.arg_count = 1;
	config.item_types[2].intensity_fn.args = (float*) malloc(sizeof(float) * 1);
	config.item_types[2].intensity_fn.args[0] = 1.5f;
	config.item_types[2].interaction_fns = (energy_function<interaction_function>*)
			malloc(sizeof(energy_function<interaction_function>) * config.item_types.length);
	config.item_types[3].intensity_fn.fn = constant_intensity_fn;
	config.item_types[3].intensity_fn.arg_count = 1;
	config.item_types[3].intensity_fn.args = (float*) malloc(sizeof(float) * 1);
	config.item_types[3].intensity_fn.args[0] = -12.0f;
	config.item_types[3].interaction_fns = (energy_function<interaction_function>*)
			malloc(sizeof(energy_function<interaction_function>) * config.item_types.length);
	config.item_types[4].intensity_fn.fn = constant_intensity_fn;
	config.item_types[4].intensity_fn.arg_count = 1;
	config.item_types[4].intensity_fn.args = (float*) malloc(sizeof(float) * 1);
	config.item_types[4].intensity_fn.args[0] = 2.0f;
	config.item_types[4].interaction_fns = (energy_function<interaction_function>*)
			malloc(sizeof(energy_function<interaction_function>) * config.item_types.length);
	config.item_types[5].intensity_fn.fn = constant_intensity_fn;
	config.item_types[5].intensity_fn.arg_count = 1;
	config.item_types[5].intensity_fn.args = (float*) malloc(sizeof(float) * 1);
	config.item_types[5].intensity_fn.args[0] = 0.0f;
	config.item_types[5].interaction_fns = (energy_function<interaction_function>*)
			malloc(sizeof(energy_function<interaction_function>) * config.item_types.length);

	set_interaction_args(config.item_types.data, 0, 0, piecewise_box_interaction_fn, {10.0f, 100.0f, 0.0f, -6.0f});
	set_interaction_args(config.item_types.data, 0, 1, zero_interaction_fn, {});
	set_interaction_args(config.item_types.data, 0, 2, piecewise_box_interaction_fn, {10.0f, 100.0f, 2.0f, -100.0f});
	set_interaction_args(config.item_types.data, 0, 3, zero_interaction_fn, {});
	set_interaction_args(config.item_types.data, 0, 4, piecewise_box_interaction_fn, {50.0f, 100.0f, -100.0f, -100.0f});
	set_interaction_args(config.item_types.data, 0, 5, zero_interaction_fn, {});

	set_interaction_args(config.item_types.data, 1, 0, zero_interaction_fn, {});
	set_interaction_args(config.item_types.data, 1, 1, zero_interaction_fn, {});
	set_interaction_args(config.item_types.data, 1, 2, zero_interaction_fn, {});
	set_interaction_args(config.item_types.data, 1, 3, zero_interaction_fn, {});
	set_interaction_args(config.item_types.data, 1, 4, zero_interaction_fn, {});
	set_interaction_args(config.item_types.data, 1, 5, zero_interaction_fn, {});

	set_interaction_args(config.item_types.data, 2, 0, piecewise_box_interaction_fn, {10.0f, 100.0f, 2.0f, -100.0f});
	set_interaction_args(config.item_types.data, 2, 1, zero_interaction_fn, {});
	set_interaction_args(config.item_types.data, 2, 2, piecewise_box_interaction_fn, {10.0f, 100.0f, 0.0f, -6.0f});
	set_interaction_args(config.item_types.data, 2, 3, zero_interaction_fn, {});
	set_interaction_args(config.item_types.data, 2, 4, piecewise_box_interaction_fn, {50.0f, 100.0f, -100.0f, -100.0f});
	set_interaction_args(config.item_types.data, 2, 5, zero_interaction_fn, {});

	set_interaction_args(config.item_types.data, 3, 0, zero_interaction_fn, {});
	set_interaction_args(config.item_types.data, 3, 1, zero_interaction_fn, {});
	set_interaction_args(config.item_types.data, 3, 2, zero_interaction_fn, {});
	set_interaction_args(config.item_types.data, 3, 3, cross_interaction_fn, {20.0f, 40.0f, 8.0f, -1000.0f, -1000.0f, -1.0f});
	set_interaction_args(config.item_types.data, 3, 4, zero_interaction_fn, {});
	set_interaction_args(config.item_types.data, 3, 5, zero_interaction_fn, {});

	set_interaction_args(config.item_types.data, 4, 0, zero_interaction_fn, {});
	set_interaction_args(config.item_types.data, 4, 1, zero_interaction_fn, {});
	set_interaction_args(config.item_types.data, 4, 2, zero_interaction_fn, {});
	set_interaction_args(config.item_types.data, 4, 3, zero_interaction_fn, {});
	set_interaction_args(config.item_types.data, 4, 4, piecewise_box_interaction_fn, {100.0f, 500.0f, 0.0f, -0.1f});
	set_interaction_args(config.item_types.data, 4, 5, zero_interaction_fn, {});

	set_interaction_args(config.item_types.data, 5, 0, zero_interaction_fn, {});
	set_interaction_args(config.item_types.data, 5, 1, zero_interaction_fn, {});
	set_interaction_args(config.item_types.data, 5, 2, zero_interaction_fn, {});
	set_interaction_args(config.item_types.data, 5, 3, zero_interaction_fn, {});
	set_interaction_args(config.item_types.data, 5, 4, piecewise_box_interaction_fn, {4.0f, 200.0f, 2.0f, 0.0f});
	set_interaction_args(config.item_types.data, 5, 5, piecewise_box_interaction_fn, {30.0f, 1000.0f, -0.3f, -1.0f});

	unsigned int jellybean_index = (unsigned int) config.item_types.length;
	unsigned int onion_index = (unsigned int) config.item_types.length;
	unsigned int wall_index = (unsigned int) config.item_types.length;
	for (unsigned int i = 0; i < config.item_types.length; i++) {
		if (config.item_types[i].name == "jellybean") {
			jellybean_index = i;
		//} else if (config.item_types[i].name == "onion") {
		//	onion_index = i;
		} else if (config.item_types[i].name == "wall") {
			wall_index = i;
		}
	}

	if (jellybean_index == config.item_types.length) {
		fprintf(stderr, "ERROR: There is no item named 'jellybean'.\n");
		return EXIT_FAILURE;
	} if (onion_index == config.item_types.length) {
		fprintf(stderr, "WARNING: There is no item named 'onion'.\n");
	} if (wall_index == config.item_types.length) {
		fprintf(stderr, "WARNING: There is no item named 'wall'.\n");
	}

	const float* jellybean_color = config.item_types[jellybean_index].color;
	float* wall_color = (float*) alloca(sizeof(float) * config.color_dimension);
	if (wall_index == config.item_types.length) {
		for (unsigned int i = 0; i < config.color_dimension; i++)
			wall_color[i] = -1.0f;
	} else {
		for (unsigned int i = 0; i < config.color_dimension; i++)
			wall_color[i] = config.item_types[wall_index].color[i];
	}

	float* onion_color = (float*) alloca(sizeof(float) * config.color_dimension);
	if (onion_index == config.item_types.length) {
		for (unsigned int i = 0; i < config.color_dimension; i++)
			onion_color[i] = -1.0f;
	} else {
		for (unsigned int i = 0; i < config.color_dimension; i++)
			onion_color[i] = config.item_types[onion_index].color[i];
	}

	simulator<server_data>& sim = *((simulator<server_data>*) alloca(sizeof(simulator<server_data>)));
	if (init(sim, config, server_data(), get_seed()) != status::OK) {
		fprintf(stderr, "ERROR: Unable to initialize simulator.\n");
		return EXIT_FAILURE;
	}

	async_server server;
	bool server_started = true;
	server_data& sim_data = sim.get_data();
	sim_data.server = &server;
	if (!init_server(server, sim, 54354, 256, 8, permissions::grant_all())) {
		fprintf(stderr, "WARNING: Unable to start server.\n");
		server_started = false;
	}

	uint64_t agent_id; agent_state* agent;
	sim.add_agent(agent_id, agent);

	shortest_path_state* best_path = nullptr;
	unsigned int current_path_position = 0; /* zero-based index of current position in `best_path` */
	unsigned int current_path_length = 0; /* number of non-null states in `best_path` */
	for (unsigned int t = 0; true; t++)
	{
		shortest_path_state* new_path = shortest_path(
				agent->current_vision, config.vision_range,
				jellybean_color, wall_color, onion_color,
				config.color_dimension,
				config.agent_field_of_view);
		bool wall_in_front = item_exists(agent->current_vision, config.vision_range, config.color_dimension, wall_color, 0, 1)
						  || item_exists(agent->current_vision, config.vision_range, config.color_dimension, onion_color, 0, 1);

/*fprintf(stderr, "t:%u,pos:", t); print(agent->current_position, stderr); fprintf(stderr, ",new_path:");
if (new_path == nullptr) {
	fprintf(stderr, "[null]\n");
} else {
	shortest_path_state* curr = new_path;
	while (curr != nullptr) {
		fprintf(stderr, "(%d,%d,dir:", curr->x, curr->y); print(curr->dir, stderr); fprintf(stderr, ")");
		curr = curr->prev;
	}
	fprintf(stderr, "\n");
}
fprintf(stderr, "wall_in_front:%s\n", wall_in_front ? "true" : "false");
wall_in_front = item_exists<true>(agent->current_vision, config.vision_range, config.color_dimension, wall_color, 0, 1);*/

		if (best_path == nullptr || (new_path != nullptr && new_path->cost < best_path->cost - current_path_position))
		{
			if (best_path != nullptr) {
				free(*best_path);
				if (best_path->reference_count == 0)
					free(best_path);
			}
			best_path = new_path;
			current_path_position = 0;
			current_path_length = 0;

			shortest_path_state* curr = new_path;
			while (curr != nullptr) {
				++current_path_length;
				curr = curr->prev;
			}
			if (new_path != nullptr) ++new_path->reference_count;
		}

		if (new_path != nullptr) {
			free(*new_path);
			if (new_path->reference_count == 0)
				free(new_path);
		}

		status action_status;
		sim_data.waiting = true;
		if (best_path == nullptr) {
			if (!wall_in_front) {
				action_status = sim.move(agent_id, direction::UP, 1);
			} else if (rand() % 2 == 0) {
				action_status = sim.turn(agent_id, direction::LEFT);
			} else {
				action_status = sim.turn(agent_id, direction::RIGHT);
			}
		} else {
			shortest_path_state* curr = best_path;
			shortest_path_state* next = nullptr;
			for (unsigned int i = current_path_length - 1; i > current_path_position; i--) {
				next = curr;
				curr = curr->prev;
			}

			int new_x, new_y;
			move_forward(curr->x, curr->y, curr->dir, new_x, new_y);
			if (new_x == next->x && new_y == next->y) {
				action_status = sim.move(agent_id, direction::UP, 1);
			} else if (next->dir == turn_left(curr->dir)) {
				action_status = sim.turn(agent_id, direction::LEFT);
			} else if (next->dir == turn_right(curr->dir)) {
				action_status = sim.turn(agent_id, direction::RIGHT);
			} else {
				fprintf(stderr, "ERROR: `shortest_path` returned an invalid path.\n");
				action_status = status::AGENT_ALREADY_ACTED;
				sim_data.waiting = false;
			}

			if (action_status == status::OK) {
				++current_path_position;
				if (current_path_position + 1 >= current_path_length) {
					free(*best_path);
					if (best_path->reference_count == 0)
						free(best_path);
					best_path = nullptr;
				}
			}
		}

		if (action_status != status::OK) t--;

		std::unique_lock<std::mutex> lock(sim_data.lock);
		while (sim_data.waiting) sim_data.cv.wait(lock);
		lock.unlock();

		if (t % 1000 == 0) {
			FILE* out = stdout;
			fprintf(out, "[iteration %u]\n"
				"  Agent position: ", t);
			double reward;
			if (onion_index < config.item_types.length) {
				reward = ((double) agent->collected_items[jellybean_index] - agent->collected_items[onion_index]) / (t + 1);
			} else {
				reward = (double) agent->collected_items[jellybean_index] / (t + 1);
			}
			print(agent->current_position, out); fprintf(out, "\n  Jellybeans collected: %u\n  Reward rate: %lf\n", agent->collected_items[jellybean_index], reward);
			fflush(out);
		}
	}

	if (best_path != nullptr) {
		free(*best_path);
		if (best_path->reference_count == 0)
			free(best_path);
	}

	if (server_started)
		stop_server(server);
	free(sim);
}
