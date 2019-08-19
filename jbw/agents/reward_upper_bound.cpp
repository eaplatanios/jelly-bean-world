#include <jbw/simulator.h>
#include <set>
#include <thread>

using namespace core;
using namespace jbw;

inline unsigned int get_distance(const unsigned int* distances,
		unsigned int start_vertex, direction start_direction,
		unsigned int end_vertex, direction end_direction,
		unsigned int vertex_count)
{
	return distances[((start_vertex * (uint_fast8_t) direction::COUNT + (uint_fast8_t) start_direction)
			* vertex_count + end_vertex) * (uint_fast8_t) direction::COUNT + (uint_fast8_t) end_direction];
}

inline unsigned int* get_row(unsigned int* distances,
		unsigned int start_vertex, direction start_direction,
		unsigned int vertex_count)
{
	return &distances[((start_vertex * (uint_fast8_t) direction::COUNT + (uint_fast8_t) start_direction) * vertex_count) * (uint_fast8_t) direction::COUNT];
}

struct fixed_length_shortest_path_state
{
	unsigned int vertex_id;
	direction dir;
	unsigned int distance;
	unsigned int length;

	struct less_than {
		inline bool operator () (const fixed_length_shortest_path_state left, const fixed_length_shortest_path_state right) {
			return left.distance < right.distance;
		}
	};
};

unsigned int fixed_length_shortest_path(
		unsigned int start_vertex, direction start_direction,
		unsigned int end_vertex, direction end_direction,
		const unsigned int* distances, unsigned int vertex_count,
		const unsigned int k, const unsigned int* disallowed_vertices,
		const unsigned int disallowed_vertex_count)
{
	unsigned int state_count = vertex_count * (k + 1) * (uint_fast8_t) direction::COUNT;
	unsigned int* smallest_costs = (unsigned int*) malloc(sizeof(unsigned int) * state_count);
	for (unsigned int i = 0; i < state_count; i++)
		smallest_costs[i] = UINT_MAX;

	std::multiset<fixed_length_shortest_path_state, fixed_length_shortest_path_state::less_than> queue;
	fixed_length_shortest_path_state initial_state;
	initial_state.vertex_id = start_vertex;
	initial_state.dir = start_direction;
	initial_state.distance = 1; /* we want to include the cost of moving into this region from the previous region */
	initial_state.length = 0;
	smallest_costs[(initial_state.vertex_id * (k + 1) + initial_state.length) * (uint_fast8_t) direction::COUNT + (uint_fast8_t) initial_state.dir] = initial_state.distance;
	queue.insert(initial_state);

	unsigned int shortest_distance = UINT_MAX;
	while (!queue.empty()) {
		auto first = queue.cbegin();
		fixed_length_shortest_path_state state = *first;
		queue.erase(first);

		if (state.vertex_id == end_vertex) {
			if (state.length != k)
				fprintf(stderr, "fixed_length_shortest_path WARNING: Completed path does not have length `k`.\n");
			if (state.dir != end_direction)
				fprintf(stderr, "fixed_length_shortest_path WARNING: Completed path does not have direction `end_direction` in the last state.\n");
			shortest_distance = state.distance;
			break;
		}

		if (state.length == k - 1) {
			/* we need the next vertex to be the `end_vertex` */
			unsigned int new_distance = state.distance + get_distance(distances, state.vertex_id, state.dir, end_vertex, end_direction, vertex_count);
			if (new_distance < smallest_costs[(end_vertex * (k + 1) + k) * (uint_fast8_t) direction::COUNT + (uint_fast8_t) end_direction]) {
				smallest_costs[(end_vertex * (k + 1) + k) * (uint_fast8_t) direction::COUNT + (uint_fast8_t) end_direction] = new_distance;

				fixed_length_shortest_path_state new_state;
				new_state.vertex_id = end_vertex;
				new_state.dir = end_direction;
				new_state.length = state.length + 1;
				new_state.distance = new_distance;
				queue.insert(new_state);
			}
		} else if (state.length >= k) {
			fprintf(stderr, "fixed_length_shortest_path WARNING: This path has length at least `k`.\n");
		} else {
			for (unsigned int i = 0; i < vertex_count; i++) {
				if (i == start_vertex || i == end_vertex || i == state.vertex_id) continue;
				if (index_of(i, disallowed_vertices, disallowed_vertex_count) < disallowed_vertex_count) continue;
				for (uint_fast8_t d = 0; d < (uint_fast8_t) direction::COUNT; d++) {
					direction dir = (direction) d;
					unsigned int new_distance = state.distance + get_distance(distances, state.vertex_id, state.dir, i, dir, vertex_count);
					if (new_distance < smallest_costs[(i * (k + 1) + state.length + 1) * (uint_fast8_t) direction::COUNT + d]) {
						smallest_costs[(i * (k + 1) + state.length + 1) * (uint_fast8_t) direction::COUNT + d] = new_distance;

						fixed_length_shortest_path_state new_state;
						new_state.vertex_id = i;
						new_state.dir = dir;
						new_state.length = state.length + 1;
						new_state.distance = new_distance;
						queue.insert(new_state);
					}
				}
			}
		}
	}

	free(smallest_costs);
	return shortest_distance;
}

struct optimal_path_state
{
	unsigned int vertex_id;
	direction dir;
	unsigned int distance;
	float priority;
	optimal_path_state* prev;

	unsigned int reference_count;

	static inline void free(optimal_path_state& state) {
		state.reference_count--;
		if (state.reference_count == 0 && state.prev != nullptr) {
			core::free(*state.prev);
			if (state.prev->reference_count == 0)
				core::free(state.prev);
		}
	}

	struct less_than {
		inline bool operator () (const optimal_path_state* left, const optimal_path_state* right) {
			return left->priority < right->priority;
		}
	};
};

inline float upper_bound(
		const unsigned int* distances,
		const unsigned int vertex_count,
		const unsigned int end_vertex_id,
		const direction end_direction,
		const optimal_path_state& new_state,
		const unsigned int* remaining_vertices,
		const unsigned int remaining_vertex_count,
		unsigned int* visited_vertices,
		unsigned int visited_vertex_count)
{
	float best_reward_rate = (float) (visited_vertex_count + 1 - 2) / new_state.distance;
	if (new_state.vertex_id == end_vertex_id) return best_reward_rate;
	for (unsigned int k = 1; k < remaining_vertex_count; k++) {
		unsigned int distance = fixed_length_shortest_path(new_state.vertex_id, new_state.dir,
				end_vertex_id, end_direction, distances, vertex_count, k + 1, visited_vertices, visited_vertex_count);
		float reward_rate = (float) (visited_vertex_count + k + 1 - 2) / (new_state.distance + distance);
		if (reward_rate > best_reward_rate)
			best_reward_rate = reward_rate;
	}
	return best_reward_rate;
}

optimal_path_state* find_optimal_path(
		const unsigned int* distances, unsigned int vertex_count,
		unsigned int start_vertex_id, direction start_direction,
		unsigned int end_vertex_id, direction end_direction)
{
	std::multiset<optimal_path_state*, optimal_path_state::less_than> queue;
	optimal_path_state* initial_state = (optimal_path_state*) malloc(sizeof(optimal_path_state));
	initial_state->vertex_id = start_vertex_id;
	initial_state->dir = start_direction;
	initial_state->distance = 1; /* we want to include the cost of moving into this region from the previous region */
	unsigned int distance_to_target = get_distance(distances, start_vertex_id, start_direction, end_vertex_id, end_direction, vertex_count);
	initial_state->priority = (float) (vertex_count - 2) / (initial_state->distance + distance_to_target);
	initial_state->prev = nullptr;
	initial_state->reference_count = 1;
	queue.insert(initial_state);

	unsigned int visited_vertex_count = 0, remaining_vertex_count = 0;
	unsigned int* visited_vertices = (unsigned int*) malloc(sizeof(unsigned int) * vertex_count);
	unsigned int* remaining_vertices = (unsigned int*) malloc(sizeof(unsigned int) * vertex_count);
	unsigned int* vertex_ids = (unsigned int*) malloc(sizeof(unsigned int) * vertex_count);
	for (unsigned int i = 0; i < vertex_count; i++)
		vertex_ids[i] = i;

	float best_score = -1.0f;
	optimal_path_state* best_path = nullptr;
	float last_priority = std::numeric_limits<float>::max();
	while (!queue.empty()) {
		auto last = queue.cend(); last--;
		optimal_path_state* state = *last;
		if (state->priority <= best_score) {
			/* the search priority is at most the best score, so we have found the optimum */
			break;
		}
		queue.erase(last);

		if (state->priority > last_priority)
			fprintf(stderr, "parse WARNING: Search is not monotonic.\n");

		visited_vertex_count = 0;
		remaining_vertex_count = 0;
		optimal_path_state* curr = state;
		while (curr != nullptr) {
			visited_vertices[visited_vertex_count++] = curr->vertex_id;
			curr = curr->prev;
		}
		insertion_sort(visited_vertices, visited_vertex_count);
		set_subtract(remaining_vertices, remaining_vertex_count,
				vertex_ids, vertex_count, visited_vertices, visited_vertex_count);

		/* check if we reached the `end_vertex_id` */
		if (state->vertex_id == end_vertex_id && state->dir == end_direction) {
			/* we reached `end_vertex_id`, so we can stop the search */
			float score = (double) (visited_vertex_count - 2) / state->distance;
			if (score > best_score) {
				if (best_path != nullptr) {
					free(*best_path);
					if (best_path->reference_count == 0)
						free(best_path);
				}

				best_path = state;
				best_path->reference_count++;
				best_score = score;
			}
		} else {
			for (unsigned int i = 0; i < remaining_vertex_count; i++) {
				unsigned int next_vertex = remaining_vertices[i];
				for (uint_fast8_t d = (uint_fast8_t) direction::UP; d < (uint_fast8_t) direction::COUNT; d++) {
					direction dir = (direction) d;
					if (next_vertex == end_vertex_id && dir != end_direction) continue;
					unsigned int next_distance = get_distance(distances, state->vertex_id, state->dir, next_vertex, dir, vertex_count);
					if (next_distance == UINT_MAX) continue;

					optimal_path_state* new_state = (optimal_path_state*) malloc(sizeof(optimal_path_state));
					new_state->vertex_id = next_vertex;
					new_state->dir = dir;
					new_state->distance = state->distance + next_distance;
					//unsigned int distance_to_target = get_distance(distances, next_vertex, dir, end_vertex_id, end_direction, vertex_count);
					//new_state->priority = (float) (visited_vertex_count + remaining_vertex_count - 2) / (new_state->distance + distance_to_target);
					new_state->priority = upper_bound(distances, vertex_count, end_vertex_id, end_direction,
							*new_state, remaining_vertices, remaining_vertex_count, visited_vertices, visited_vertex_count);
					new_state->prev = state;
					new_state->reference_count = 1;
					++state->reference_count;
					queue.insert(new_state);
				}
			}
		}

		/* remove states from the queue with strictly worse bound than the best score we have so far */
		/*while (!queue.empty()) {
			auto first = queue.cbegin();
			optimal_path_state* state = *first;
			if (state->priority >= best_score)
				break;
			queue.erase(first);
			free(*state);
			if (state->reference_count == 0)
				free(state);
		}*/

		free(*state);
		if (state->reference_count == 0)
			free(state);
	}

	for (auto state : queue) {
		free(*state);
		if (state->reference_count == 0)
			free(state);
	}
	free(visited_vertices);
	free(remaining_vertices);
	free(vertex_ids);
	return best_path;
}

struct shortest_path_state
{
	unsigned int cost;
	unsigned int x, y;
	direction dir;

	struct less_than {
		inline bool operator () (const shortest_path_state left, const shortest_path_state right) {
			return left.cost < right.cost;
		}
	};
};

inline void move_forward(
		unsigned int x, unsigned int y, direction dir,
		unsigned int max_x, unsigned int max_y,
		unsigned int& new_x, unsigned int& new_y)
{
	new_x = x;
	new_y = y;
	if (dir == direction::UP) {
		++new_y;
		if (new_y > max_y) new_y = UINT_MAX;
	} else if (dir == direction::DOWN) {
		if (new_y == 0) new_y = UINT_MAX;
		else --new_y;
	} else if (dir == direction::LEFT) {
		if (new_x == 0) new_x = UINT_MAX;
		else --new_x;
	} else if (dir == direction::RIGHT) {
		++new_x;
		if (new_x > max_x) new_x = UINT_MAX;
	}
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

void compute_shortest_distances(
		unsigned int start_x, unsigned int start_y,
		direction start_direction,
		unsigned int max_x, unsigned int max_y,
		const array<position>& goals,
		const array<position>& walls,
		unsigned int* shortest_distances)
{
	unsigned int state_count = (max_y + 1) * (max_x + 1) * (uint_fast8_t) direction::COUNT;
	unsigned int* smallest_costs = (unsigned int*) malloc(sizeof(unsigned int) * state_count);
	for (unsigned int i = 0; i < state_count; i++)
		smallest_costs[i] = UINT_MAX;
	for (unsigned int i = 0; i < (goals.length + 1) * (uint_fast8_t) direction::COUNT; i++)
		shortest_distances[i] = UINT_MAX;

	std::multiset<shortest_path_state, shortest_path_state::less_than> queue;
	shortest_path_state initial_state;
	initial_state.cost = 0;
	initial_state.x = start_x;
	initial_state.y = start_y;
	initial_state.dir = start_direction;
	smallest_costs[(start_x * (max_y + 1) + start_y) * (uint_fast8_t) direction::COUNT + (uint_fast8_t) start_direction] = 0;
	queue.insert(initial_state);

	while (!queue.empty()) {
		auto first = queue.cbegin();
		shortest_path_state state = *first;
		queue.erase(first);

		/* check if we found a jellybean */
		unsigned int goal_index = goals.index_of({state.x, state.y});
		if (goal_index < goals.length) {
			/* we found a jellybean */
			shortest_distances[goal_index * (uint_fast8_t) direction::COUNT + (uint_fast8_t) state.dir] = state.cost;
		} if (state.y == max_y) {
			/* we reached the top row */
			shortest_distances[goals.length * (uint_fast8_t) direction::COUNT + (uint_fast8_t) state.dir] =
					min(shortest_distances[goals.length * (uint_fast8_t) direction::COUNT + (uint_fast8_t) state.dir], state.cost);
		}

		/* consider moving forward */
		unsigned int new_x, new_y;
		direction new_dir = state.dir;
		move_forward(state.x, state.y, state.dir, max_x, max_y, new_x, new_y);
		if (new_x != UINT_MAX && new_y != UINT_MAX) {
			/* check if there is a wall in the new position */
			if (!walls.contains({new_x, new_y})) {
				/* there is no wall, so continue considering this movement */
				unsigned int new_cost = state.cost + 1;
				if (new_cost < smallest_costs[(new_x * (max_y + 1) + new_y) * (uint_fast8_t) direction::COUNT + (uint_fast8_t) new_dir]) {
					smallest_costs[(new_x * (max_y + 1) + new_y) * (uint_fast8_t) direction::COUNT + (uint_fast8_t) new_dir] = new_cost;

					shortest_path_state new_state;
					new_state.cost = new_cost;
					new_state.x = new_x;
					new_state.y = new_y;
					new_state.dir = new_dir;
					queue.insert(new_state);
				}
			}
		}

		/* consider turning left */
		unsigned int new_cost = state.cost + 1;
		new_x = state.x;
		new_y = state.y;
		new_dir = turn_left(state.dir);
		if (new_cost < smallest_costs[(new_x * (max_y + 1) + new_y) * (uint_fast8_t) direction::COUNT + (uint_fast8_t) new_dir]) {
			smallest_costs[(new_x * (max_y + 1) + new_y) * (uint_fast8_t) direction::COUNT + (uint_fast8_t) new_dir] = new_cost;

			shortest_path_state new_state;
			new_state.cost = new_cost;
			new_state.x = new_x;
			new_state.y = new_y;
			new_state.dir = new_dir;
			queue.insert(new_state);
		}

		/* consider turning right */
		new_cost = state.cost + 1;
		new_x = state.x;
		new_y = state.y;
		new_dir = turn_right(state.dir);
		if (new_cost < smallest_costs[(new_x * (max_y + 1) + new_y) * (uint_fast8_t) direction::COUNT + (uint_fast8_t) new_dir]) {
			smallest_costs[(new_x * (max_y + 1) + new_y) * (uint_fast8_t) direction::COUNT + (uint_fast8_t) new_dir] = new_cost;

			shortest_path_state new_state;
			new_state.cost = new_cost;
			new_state.x = new_x;
			new_state.y = new_y;
			new_state.dir = new_dir;
			queue.insert(new_state);
		}
	}

	free(smallest_costs);
}

struct empty_data {
	static inline void move(const empty_data& src, empty_data& dst) { }
	static inline void free(empty_data& data) { }
};

constexpr bool init(empty_data& data) { return true; }

template<typename PerPatchData, typename ItemType>
void generate_map(
		map<PerPatchData, ItemType>& world,
		const position& bottom_left_corner,
		const position& top_right_corner)
{
	/* make sure enough of the world is generated */
	patch<empty_data>* neighborhood[4]; position patch_positions[4];
	for (int64_t x = bottom_left_corner.x; x <= top_right_corner.x; x += world.n) {
		for (int64_t y = bottom_left_corner.y; y <= top_right_corner.y; y += world.n)
			world.get_fixed_neighborhood(position(x, y), neighborhood, patch_positions);
		world.get_fixed_neighborhood(position(x, top_right_corner.y), neighborhood, patch_positions);
	}
	for (int64_t y = bottom_left_corner.y; y <= top_right_corner.y; y += world.n)
		world.get_fixed_neighborhood(position(top_right_corner.x, y), neighborhood, patch_positions);
	world.get_fixed_neighborhood(top_right_corner, neighborhood, patch_positions);
}

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

inline void compute_optimal_reward_rate(
		const unsigned int worker_id,
		const unsigned int n,
		const unsigned int mcmc_iterations,
		const item_properties* item_types,
		const unsigned int item_type_count,
		const unsigned int jellybean_index,
		position bottom_left_corner,
		position top_right_corner,
		position agent_start_position,
		std::mutex& lock, std::minstd_rand& rng,
		array<float>& reward_rates)
{
	lock.lock();
	unsigned int seed = rng();
	lock.unlock();
	auto m = map<empty_data, item_properties>(n, mcmc_iterations, item_types, item_type_count, seed);
	generate_map(m, bottom_left_corner, top_right_corner);

	position bottom_left_patch_position, top_right_patch_position;
	m.world_to_patch_coordinates(bottom_left_corner, bottom_left_patch_position);
	m.world_to_patch_coordinates(top_right_corner, top_right_patch_position);
	array<position> goals(64), walls(64);
	apply_contiguous(m.patches, bottom_left_patch_position.y,
		top_right_patch_position.y - bottom_left_patch_position.y + 1,
		[&](const array_map<int64_t, patch<empty_data>>& row, int64_t y)
	{
		return apply_contiguous(row, bottom_left_patch_position.x,
			top_right_patch_position.x - bottom_left_patch_position.x + 1,
			[&](const patch<empty_data>& patch, int64_t x)
		{
			for (const item& i : patch.items) {
				if (i.location.x >= bottom_left_corner.x && i.location.x <= top_right_corner.x
				&& i.location.y >= bottom_left_corner.y && i.location.y <= top_right_corner.y)
				{
					if (item_types[i.item_type].blocks_movement) walls.add(i.location);
					else if (i.item_type == jellybean_index) goals.add(i.location);
				}
			}
			return true;
		});
	});

	/* the first vertex is the agent starting position, and the last vertex is the upper row of the map region */
	unsigned int* distances = (unsigned int*) malloc(sizeof(unsigned int) * (goals.length + 2) * (goals.length + 2) * (uint_fast8_t) direction::COUNT * (uint_fast8_t) direction::COUNT);
	for (uint_fast8_t d = 0; d < (uint_fast8_t) direction::COUNT; d++) {
		unsigned int* row = get_row(distances, 0, (direction) d, goals.length + 2);
		for (uint_fast8_t e = 0; e < (uint_fast8_t) direction::COUNT; e++) {
			if (e == d) {
				row[e] = 0;
			} else if ((direction) e == turn_left((direction) d) || (direction) e == turn_right((direction) d)) {
				row[e] = 1;
			} else {
				row[e] = 2;
			}
		}

		if ((direction) d != direction::UP) {
			for (unsigned int i = 0; i < (goals.length + 1) * (uint_fast8_t) direction::COUNT; i++)
				row[(uint_fast8_t) direction::COUNT + i] = UINT_MAX;
		}
	}
	for (uint_fast8_t d = 0; d < (uint_fast8_t) direction::COUNT; d++) {
		unsigned int* row = get_row(distances, goals.length + 1, (direction) d, goals.length + 2);
		for (uint_fast8_t e = 0; e < (uint_fast8_t) direction::COUNT; e++) {
			if (e == d) {
				row[(goals.length + 1) * (uint_fast8_t) direction::COUNT + e] = 0;
			} else if ((direction) e == turn_left((direction) d) || (direction) e == turn_right((direction) d)) {
				row[(goals.length + 1) * (uint_fast8_t) direction::COUNT + e] = 1;
			} else {
				row[(goals.length + 1) * (uint_fast8_t) direction::COUNT + e] = 2;
			}
		}

		if ((direction) d != direction::UP) {
			for (unsigned int i = 0; i < (goals.length + 1) * (uint_fast8_t) direction::COUNT; i++)
				row[i] = UINT_MAX;
		}
	}
	compute_shortest_distances(agent_start_position.x, agent_start_position.y,
			direction::UP, top_right_corner.x, top_right_corner.y, goals, walls,
			distances + (uint_fast8_t) direction::COUNT);
	for (unsigned int i = 0; i < goals.length; i++) {
		for (uint_fast8_t d = 0; d < (uint_fast8_t) direction::COUNT; d++) {
			unsigned int* row = get_row(distances, i + 1, (direction) d, goals.length + 2);
			for (uint_fast8_t e = 0; e < (uint_fast8_t) direction::COUNT; e++)
				row[e] = UINT_MAX; /* don't allow movement back to the agent "vertex" */
			compute_shortest_distances(goals[i].x, goals[i].y,
					(direction) d, top_right_corner.x, top_right_corner.y,
					goals, walls, row + (uint_fast8_t) direction::COUNT);
		}
	}

	fprintf(stderr, "[thread %u] Finding optimal path with jellybean count: %zu\n", worker_id, goals.length);
	optimal_path_state* path = find_optimal_path(distances, goals.length + 2, 0, direction::UP, goals.length + 1, direction::UP);

	unsigned int path_length = 0; /* in vertices, including endpoints */
	optimal_path_state* curr = path;
	while (curr != nullptr) {
		path_length++;
		curr = curr->prev;
	}

	lock.lock();
	reward_rates.add((float) (path_length - 2) / path->distance);
	float mean = 0.0f;
	for (float x : reward_rates)
		mean += x;
	mean /= reward_rates.length;
	float variance = 0.0f;
	for (float x : reward_rates)
		variance += (x - mean) * (x - mean);
	variance /= (reward_rates.length + 1);
	fprintf(stderr, "Avg reward rate: %f, stddev reward rate: %f, stddev of avg: %f\n", mean, sqrt(variance), sqrt(variance / reward_rates.length));
	lock.unlock();
}

int main(int argc, const char** argv)
{
	static constexpr int n = 32;
	static constexpr unsigned int item_type_count = 4;
	static constexpr unsigned int mcmc_iterations = 4000;
	static constexpr unsigned int scent_dimension = 3;
	static constexpr unsigned int color_dimension = 3;
	item_properties* item_types = (item_properties*) alloca(sizeof(item_properties) * item_type_count);

	item_types[0].name = "banana";
	item_types[0].scent = (float*) calloc(scent_dimension, sizeof(float));
	item_types[0].color = (float*) calloc(color_dimension, sizeof(float));
	item_types[0].required_item_counts = (unsigned int*) calloc(item_type_count, sizeof(unsigned int));
	item_types[0].required_item_costs = (unsigned int*) calloc(item_type_count, sizeof(unsigned int));
	item_types[0].scent[1] = 1.0f;
	item_types[0].color[1] = 1.0f;
	item_types[0].required_item_counts[0] = 1;
	item_types[0].blocks_movement = false;
	item_types[0].visual_occlusion = 0.0;
	item_types[1].name = "onion";
	item_types[1].scent = (float*) calloc(scent_dimension, sizeof(float));
	item_types[1].color = (float*) calloc(color_dimension, sizeof(float));
	item_types[1].required_item_counts = (unsigned int*) calloc(item_type_count, sizeof(unsigned int));
	item_types[1].required_item_costs = (unsigned int*) calloc(item_type_count, sizeof(unsigned int));
	item_types[1].scent[0] = 1.0f;
	item_types[1].color[0] = 1.0f;
	item_types[1].required_item_counts[1] = 1;
	item_types[1].blocks_movement = false;
	item_types[1].visual_occlusion = 0.0;
	item_types[2].name = "jellybean";
	item_types[2].scent = (float*) calloc(scent_dimension, sizeof(float));
	item_types[2].color = (float*) calloc(color_dimension, sizeof(float));
	item_types[2].required_item_counts = (unsigned int*) calloc(item_type_count, sizeof(unsigned int));
	item_types[2].required_item_costs = (unsigned int*) calloc(item_type_count, sizeof(unsigned int));
	item_types[2].scent[2] = 1.0f;
	item_types[2].color[2] = 1.0f;
	item_types[2].blocks_movement = false;
	item_types[2].visual_occlusion = 0.0;
	item_types[3].name = "wall";
	item_types[3].scent = (float*) calloc(scent_dimension, sizeof(float));
	item_types[3].color = (float*) calloc(color_dimension, sizeof(float));
	item_types[3].required_item_counts = (unsigned int*) calloc(item_type_count, sizeof(unsigned int));
	item_types[3].required_item_costs = (unsigned int*) calloc(item_type_count, sizeof(unsigned int));
	item_types[3].color[0] = 0.52f;
	item_types[3].color[1] = 0.22f;
	item_types[3].color[2] = 0.16f;
	item_types[3].required_item_counts[3] = 1;
	item_types[3].blocks_movement = false;
	item_types[3].visual_occlusion = 1.0f;

	item_types[0].intensity_fn.fn = constant_intensity_fn;
	item_types[0].intensity_fn.arg_count = 1;
	item_types[0].intensity_fn.args = (float*) malloc(sizeof(float) * 1);
	item_types[0].intensity_fn.args[0] = -5.3f;
	item_types[0].interaction_fns = (energy_function<interaction_function>*)
			malloc(sizeof(energy_function<interaction_function>) * item_type_count);
	item_types[1].intensity_fn.fn = constant_intensity_fn;
	item_types[1].intensity_fn.arg_count = 1;
	item_types[1].intensity_fn.args = (float*) malloc(sizeof(float) * 1);
	item_types[1].intensity_fn.args[0] = -5.0f;
	item_types[1].interaction_fns = (energy_function<interaction_function>*)
			malloc(sizeof(energy_function<interaction_function>) * item_type_count);
	item_types[2].intensity_fn.fn = constant_intensity_fn;
	item_types[2].intensity_fn.arg_count = 1;
	item_types[2].intensity_fn.args = (float*) malloc(sizeof(float) * 1);
	item_types[2].intensity_fn.args[0] = -5.3f;
	item_types[2].interaction_fns = (energy_function<interaction_function>*)
			malloc(sizeof(energy_function<interaction_function>) * item_type_count);
	item_types[3].intensity_fn.fn = constant_intensity_fn;
	item_types[3].intensity_fn.arg_count = 1;
	item_types[3].intensity_fn.args = (float*) malloc(sizeof(float) * 1);
	item_types[3].intensity_fn.args[0] = 0.0f;
	item_types[3].interaction_fns = (energy_function<interaction_function>*)
			malloc(sizeof(energy_function<interaction_function>) * item_type_count);

	set_interaction_args(item_types, 0, 0, piecewise_box_interaction_fn, {10.0f, 200.0f, 0.0f, -6.0f});
	set_interaction_args(item_types, 0, 1, piecewise_box_interaction_fn, {200.0f, 0.0f, -6.0f, -6.0f});
	set_interaction_args(item_types, 0, 2, piecewise_box_interaction_fn, {10.0f, 200.0f, 2.0f, -100.0f});
	set_interaction_args(item_types, 0, 3, zero_interaction_fn, {});
	set_interaction_args(item_types, 1, 0, piecewise_box_interaction_fn, {200.0f, 0.0f, -6.0f, -6.0f});
	set_interaction_args(item_types, 1, 1, zero_interaction_fn, {});
	set_interaction_args(item_types, 1, 2, piecewise_box_interaction_fn, {200.0f, 0.0f, -100.0f, -100.0f});
	set_interaction_args(item_types, 1, 3, zero_interaction_fn, {});
	set_interaction_args(item_types, 2, 0, piecewise_box_interaction_fn, {10.0f, 200.0f, 2.0f, -100.0f});
	set_interaction_args(item_types, 2, 1, piecewise_box_interaction_fn, {200.0f, 0.0f, -100.0f, -100.0f});
	set_interaction_args(item_types, 2, 2, piecewise_box_interaction_fn, {10.0f, 200.0f, 0.0f, -6.0f});
	set_interaction_args(item_types, 2, 3, zero_interaction_fn, {});
	set_interaction_args(item_types, 3, 0, zero_interaction_fn, {});
	set_interaction_args(item_types, 3, 1, zero_interaction_fn, {});
	set_interaction_args(item_types, 3, 2, zero_interaction_fn, {});
	set_interaction_args(item_types, 3, 3, cross_interaction_fn, {10.0f, 15.0f, 20.0f, -200.0f, -20.0f, 1.0f});

	unsigned int jellybean_index = item_type_count;
	for (unsigned int i = 0; i < item_type_count; i++) {
		if (item_types[i].name == "jellybean") {
			jellybean_index = i;
			break;
		}
	}

	if (jellybean_index == item_type_count) {
		fprintf(stderr, "ERROR: There is no item named 'jellybean'.\n");
		return false;
	}

	position bottom_left_corner = {0, 0};
	position top_right_corner = {32, 32};
	position agent_start_position = {top_right_corner.x / 2, bottom_left_corner.y};

#if defined(NDEBUG)
	unsigned int seed = milliseconds();
#else
	unsigned int seed = 0;
#endif
	std::minstd_rand rng(seed);
	std::mutex lock;
	array<float> reward_rates(512);
	constexpr unsigned int thread_count = 8;
	std::thread workers[thread_count];
	for (unsigned int i = 0; i < thread_count; i++)
		workers[i] = std::thread([&,i]() {
			while (true) {
				compute_optimal_reward_rate(i, n, mcmc_iterations, item_types, item_type_count, jellybean_index,
						bottom_left_corner, top_right_corner, agent_start_position, lock, rng, reward_rates);
			}
		});

	for (unsigned int i = 0; i < thread_count; i++) {
		if (workers[i].joinable()) {
			try {
				workers[i].join();
			} catch (...) { }
		}
	}

	for (unsigned int i = 0; i < item_type_count; i++)
		free(item_types[i], item_type_count);
	return EXIT_SUCCESS;
}
