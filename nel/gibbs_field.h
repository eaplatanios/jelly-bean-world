#ifndef NEL_GIBBS_FIELD_H_
#define NEL_GIBBS_FIELD_H_

#include <core/random.h>
#include <math/log.h>
#include "position.h"
#include "energy_functions.h"

namespace nel {

using namespace core;

/**
 * Structure for optimizing gibbs_field sampling when the intensity and/or
 * interaction functions are stationary.
 */
template<typename ItemType>
struct gibbs_field_cache
{
	float* intensities;
	float** interactions;
	unsigned int two_n, four_n;

	const ItemType* item_types;
	unsigned int item_type_count;

	gibbs_field_cache(const ItemType* item_types, unsigned int item_type_count, unsigned int n) :
		two_n(2*n), four_n(4*n), item_types(item_types), item_type_count(item_type_count)
	{
		if (!init_helper()) exit(EXIT_FAILURE);
	}

	~gibbs_field_cache() { free_helper(); }

	inline float intensity(const position& pos, unsigned int item_type) {
		if (is_stationary(item_types[item_type].intensity_fn.fn))
			return intensities[item_type];
		else return item_types[item_type].intensity_fn.fn(pos, item_types[item_type].intensity_fn.args);
	}

	inline float interaction(
			const position& first_position, const position& second_position,
			unsigned int first_item_type, unsigned int second_item_type)
	{
		interaction_function interaction = item_types[first_item_type].interaction_fns[second_item_type].fn;
		if (is_constant(interaction) || !is_stationary(interaction)) {
			if (first_position == second_position) return 0.0f;
			return interaction(first_position, second_position, item_types[first_item_type].interaction_fns[second_item_type].args);
		} else {
			position diff = first_position - second_position + position(two_n, two_n);
#if !defined(NDEBUG)
			if (diff.x < 0 || diff.x >= four_n || diff.y < 0 || diff.y >= four_n) {
				fprintf(stderr, "gibbs_field_cache.interaction WARNING: The "
						"given two positions further than 4*n from each other.");
				return 0.0f;
			}
#endif
			return interactions[first_item_type*item_type_count + second_item_type][diff.x*four_n + diff.y];
		}
	}

	static inline void free(gibbs_field_cache& cache) { cache.free_helper(); }

private:
	inline bool init_helper() {
		intensities = (float*) malloc(sizeof(float) * item_type_count);
		if (intensities == NULL) {
			fprintf(stderr, "gibbs_field_cache.init_helper ERROR: Insufficient memory for intensities.\n");
			return false;
		}
		interactions = (float**) calloc(item_type_count * item_type_count, sizeof(float*));
		if (interactions == NULL) {
			fprintf(stderr, "gibbs_field_cache.init_helper ERROR: Insufficient memory for interactions.\n");
			core::free(intensities);
			return false;
		}
		for (unsigned int i = 0; i < item_type_count; i++) {
			if (is_stationary(item_types[i].intensity_fn.fn))
				intensities[i] = item_types[i].intensity_fn.fn(position(0, 0), item_types[i].intensity_fn.args);

			for (unsigned int j = 0; j < item_type_count; j++) {
				interaction_function interaction = item_types[i].interaction_fns[j].fn;
				if (!is_constant(interaction) && is_stationary(interaction)) {
					interactions[i*item_type_count + j] = (float*) malloc(sizeof(float) * four_n * four_n);
					if (interactions[i*item_type_count + j] == NULL) {
						fprintf(stderr, "gibbs_field_cache.init_helper ERROR: Insufficient memory for interactions.\n");
						free_helper(); return false;
					}

					for (unsigned int x = 0; x < four_n; x++) {
						for (unsigned int y = 0; y < four_n; y++) {
							float value;
							if (x == two_n && y == two_n)
								value = 0.0f;
							else value = interaction(position(two_n, two_n), position(x, y), item_types[i].interaction_fns[j].args);
							interactions[i*item_type_count + j][x*four_n + y] = value;
						}
					}
				} else {
					interactions[i*item_type_count + j] = NULL;
				}
			}
		}
		return true;
	}

	inline void free_helper() {
		core::free(intensities);
		for (unsigned int i = 0; i < item_type_count * item_type_count; i++)
			if (interactions[i] != NULL) core::free(interactions[i]);
		core::free(interactions);
	}

	template<typename A>
	friend bool init(gibbs_field_cache<A>&, const A*, unsigned int, unsigned int);
};

template<typename ItemType>
bool init(gibbs_field_cache<ItemType>& cache,
		const ItemType* item_types, unsigned int item_type_count, unsigned int n)
{
	cache.two_n = 2*n;
	cache.four_n = 4*n;
	cache.item_types = item_types;
	cache.item_type_count = item_type_count;
	return cache.init_helper();
}

template<typename Map>
class gibbs_field
{
	typedef typename Map::patch_type patch_type;
	typedef typename Map::item_type item_type;

	Map& map;
	gibbs_field_cache<item_type>& cache;
	position* patch_positions;
	unsigned int patch_count;

	unsigned int n;

public:
	/* NOTE: `patch_positions` is used directly, and not copied, so the caller maintains ownership */
	gibbs_field(Map& map, gibbs_field_cache<item_type>& cache, position* patch_positions,
			unsigned int patch_count, unsigned int n) :
		map(map), cache(cache), patch_positions(patch_positions), patch_count(patch_count), n(n) { }

	~gibbs_field() { }

	template<typename RNGType>
	void sample(RNGType& rng) {
		for (unsigned int i = 0; i < patch_count; i++) {
			patch_type* current = map.get_patch_if_exists(patch_positions[i]);
			const patch_type* top = map.get_patch_if_exists(patch_positions[i].up());
			const patch_type* bottom = map.get_patch_if_exists(patch_positions[i].down());
			const patch_type* left = map.get_patch_if_exists(patch_positions[i].left());
			const patch_type* right = map.get_patch_if_exists(patch_positions[i].right());
			const patch_type* top_left = map.get_patch_if_exists(patch_positions[i].up().left());
			const patch_type* top_right = map.get_patch_if_exists(patch_positions[i].up().right());
			const patch_type* bottom_left = map.get_patch_if_exists(patch_positions[i].down().left());
			const patch_type* bottom_right = map.get_patch_if_exists(patch_positions[i].down().right());

			const patch_type* bottom_left_neighborhood[4];
			const patch_type* top_left_neighborhood[4];
			const patch_type* bottom_right_neighborhood[4];
			const patch_type* top_right_neighborhood[4];
			unsigned int bottom_left_neighbor_count = 1;
			unsigned int top_left_neighbor_count = 1;
			unsigned int bottom_right_neighbor_count = 1;
			unsigned int top_right_neighbor_count = 1;
			bottom_left_neighborhood[0] = current;
			top_left_neighborhood[0] = current;
			bottom_right_neighborhood[0] = current;
			top_right_neighborhood[0] = current;
			if (left != NULL) {
				bottom_left_neighborhood[bottom_left_neighbor_count++] = left;
				top_left_neighborhood[top_left_neighbor_count++] = left;
			} if (right != NULL) {
				bottom_right_neighborhood[bottom_right_neighbor_count++] = right;
				top_right_neighborhood[top_right_neighbor_count++] = right;
			} if (top != NULL) {
				top_left_neighborhood[top_left_neighbor_count++] = top;
				top_right_neighborhood[top_right_neighbor_count++] = top;
			} if (bottom != NULL) {
				bottom_left_neighborhood[bottom_left_neighbor_count++] = bottom;
				bottom_right_neighborhood[bottom_right_neighbor_count++] = bottom;
			} if (bottom_left != NULL) {
				bottom_left_neighborhood[bottom_left_neighbor_count++] = bottom_left;
			} if (top_left != NULL) {
				top_left_neighborhood[top_left_neighbor_count++] = top_left;
			} if (bottom_right != NULL) {
				bottom_right_neighborhood[bottom_right_neighbor_count++] = bottom_right;
			} if (top_right != NULL) {
				top_right_neighborhood[top_right_neighbor_count++] = top_right;
			}

			/* propose moving each item */
			const position patch_position_offset = patch_positions[i] * n;
			for (unsigned int i = 0; i < current->items.length; i++) {
fprintf(stderr, "gibbs_field.sample: proposing item movement %u\n", i);
				/* propose a new position for this item */
				const unsigned int item_type = current->items[i].item_type;
				const position old_position = current->items[i].location;
				position new_position = patch_position_offset + position(rng() % n, rng() % n);

				const patch_type** old_neighborhood;
				unsigned int old_neighborhood_size;
				if (old_position.x - patch_position_offset.x < n / 2) {
					if (old_position.y - patch_position_offset.y < n / 2) {
						old_neighborhood = bottom_left_neighborhood;
						old_neighborhood_size = bottom_left_neighbor_count;
					} else {
						old_neighborhood = top_left_neighborhood;
						old_neighborhood_size = top_left_neighbor_count;
					}
				} else {
					if (old_position.y - patch_position_offset.y < n / 2) {
						old_neighborhood = bottom_right_neighborhood;
						old_neighborhood_size = bottom_right_neighbor_count;
					} else {
						old_neighborhood = top_right_neighborhood;
						old_neighborhood_size = top_right_neighbor_count;
					}
				}

				const patch_type** new_neighborhood;
				unsigned int new_neighborhood_size;
				if (new_position.x - patch_position_offset.x < n / 2) {
					if (new_position.y - patch_position_offset.y < n / 2) {
						new_neighborhood = bottom_left_neighborhood;
						new_neighborhood_size = bottom_left_neighbor_count;
					} else {
						new_neighborhood = top_left_neighborhood;
						new_neighborhood_size = top_left_neighbor_count;
					}
				} else {
					if (new_position.y - patch_position_offset.y < n / 2) {
						new_neighborhood = bottom_right_neighborhood;
						new_neighborhood_size = bottom_right_neighbor_count;
					} else {
						new_neighborhood = top_right_neighborhood;
						new_neighborhood_size = top_right_neighbor_count;
					}
				}

				/* compute the log acceptance probability */
				float log_acceptance_probability = 0.0f;
				bool new_position_occupied = false;
				for (unsigned int j = 0; j < new_neighborhood_size; j++) {
					const auto& items = new_neighborhood[j]->items;
					for (unsigned int m = 0; m < items.length; m++) {
						if (items[m].location == new_position) {
							/* an item already exists at this proposed location */
							new_position_occupied = true; break;
						}
						log_acceptance_probability += cache.interaction(new_position, items[m].location, item_type, items[m].item_type);
						log_acceptance_probability += cache.interaction(items[m].location, new_position, items[m].item_type, item_type);
					}
					if (new_position_occupied) break;
					log_acceptance_probability -= cache.interaction(new_position, new_position, item_type, item_type);
				}
				if (new_position_occupied) {
fprintf(stderr, "gibbs_field.sample: done proposing item movement %u\n", i);
					continue;
				}

				for (unsigned int j = 0; j < old_neighborhood_size; j++) {
					const auto& items = old_neighborhood[j]->items;
					for (unsigned int m = 0; m < items.length; m++) {
						log_acceptance_probability -= cache.interaction(old_position, items[m].location, item_type, items[m].item_type);
						log_acceptance_probability -= cache.interaction(items[m].location, old_position, items[m].item_type, item_type);
					}
					log_acceptance_probability += cache.interaction(old_position, old_position, item_type, item_type);
				}

				log_acceptance_probability += cache.intensity(new_position, item_type) - cache.intensity(old_position, item_type);

				/* accept or reject the proposal depending on the computed probability */
				float random = (float) rng() / rng.max();
				if (log(random) < log_acceptance_probability) {
					/* accept the proposal */
					current->items.remove(i);
					current->items.add({item_type, new_position, 0, 0});
				}
fprintf(stderr, "gibbs_field.sample: done proposing item movement %u\n", i);
			}

			/* propose creating a new item */
fprintf(stderr, "gibbs_field.sample: proposing item creation\n");
			const unsigned int item_type = rng() % cache.item_type_count;
			position new_position = patch_position_offset + position(rng() % n, rng() % n);

			const patch_type** new_neighborhood;
			unsigned int new_neighborhood_size;
			if (new_position.x - patch_position_offset.x < n / 2) {
				if (new_position.y - patch_position_offset.y < n / 2) {
					new_neighborhood = bottom_left_neighborhood;
					new_neighborhood_size = bottom_left_neighbor_count;
				} else {
					new_neighborhood = top_left_neighborhood;
					new_neighborhood_size = top_left_neighbor_count;
				}
			} else {
				if (new_position.y - patch_position_offset.y < n / 2) {
					new_neighborhood = bottom_right_neighborhood;
					new_neighborhood_size = bottom_right_neighbor_count;
				} else {
					new_neighborhood = top_right_neighborhood;
					new_neighborhood_size = top_right_neighbor_count;
				}
			}

			float log_acceptance_probability = 0.0f;
			bool new_position_occupied = false;
			for (unsigned int j = 0; j < new_neighborhood_size; j++) {
				const auto& items = new_neighborhood[j]->items;
				for (unsigned int m = 0; m < items.length; m++) {
					if (items[m].location == new_position) {
						/* an item already exists at this proposed location */
						new_position_occupied = true; break;
					}
					log_acceptance_probability += cache.interaction(new_position, items[m].location, item_type, items[m].item_type);
					log_acceptance_probability += cache.interaction(items[m].location, new_position, items[m].item_type, item_type);
				}
				if (new_position_occupied) break;
				log_acceptance_probability -= cache.interaction(new_position, new_position, item_type, item_type);
			}
			static const float LOG_ITEM_TYPE_COUNT = log(cache.item_type_count); /* TODO: precompute this elsewhere (maybe `map`) and pass it */
			if (!new_position_occupied) {
				log_acceptance_probability += cache.intensity(new_position, item_type);

				/* add log probability of inverse proposal */
				log_acceptance_probability += -log(current->items.length + 1); /* TODO: using `log_cache` may speed this up */

				/* subtract log probability of forward proposal */
				log_acceptance_probability -= -LOG_ITEM_TYPE_COUNT - log(n*n - current->items.length); /* TODO: we can precompute log(cache.item_type_count)` */

				/* accept or reject the proposal depending on the computed probability */
				float random = (float) rng() / rng.max();
				if (log(random) < log_acceptance_probability) {
					/* accept the proposal */
					current->items.add({item_type, new_position, 0, 0});
				}
			}
fprintf(stderr, "gibbs_field.sample: done proposing item creation\n");

			/* propose deleting an item */
			if (current->items.length > 0) {
fprintf(stderr, "gibbs_field.sample: proposing item deletion\n");
				unsigned int item_index = rng() % current->items.length;
				const unsigned int old_item_type = current->items[item_index].item_type;
				const position old_position = current->items[item_index].location;

				const patch_type** old_neighborhood;
				unsigned int old_neighborhood_size;
				if (old_position.x - patch_position_offset.x < n / 2) {
					if (old_position.y - patch_position_offset.y < n / 2) {
						old_neighborhood = bottom_left_neighborhood;
						old_neighborhood_size = bottom_left_neighbor_count;
					} else {
						old_neighborhood = top_left_neighborhood;
						old_neighborhood_size = top_left_neighbor_count;
					}
				} else {
					if (old_position.y - patch_position_offset.y < n / 2) {
						old_neighborhood = bottom_right_neighborhood;
						old_neighborhood_size = bottom_right_neighbor_count;
					} else {
						old_neighborhood = top_right_neighborhood;
						old_neighborhood_size = top_right_neighbor_count;
					}
				}

				log_acceptance_probability = 0.0f;
				for (unsigned int j = 0; j < old_neighborhood_size; j++) {
					const auto& items = old_neighborhood[j]->items;
					for (unsigned int m = 0; m < items.length; m++) {
						log_acceptance_probability -= cache.interaction(old_position, items[m].location, old_item_type, items[m].item_type);
						log_acceptance_probability -= cache.interaction(items[m].location, old_position, items[m].item_type, old_item_type);
					}
					log_acceptance_probability += cache.interaction(old_position, old_position, old_item_type, old_item_type);
				}
				log_acceptance_probability -= cache.intensity(old_position, old_item_type);

				/* add log probability of inverse proposal */
				log_acceptance_probability += -LOG_ITEM_TYPE_COUNT - log(n*n - current->items.length + 1);

				/* subtract log probability of forward proposal */
				log_acceptance_probability -= -log(current->items.length); /* TODO: using `log_cache` may speed this up */

				/* accept or reject the proposal depending on the computed probability */
				float random = (float) rng() / rng.max();
				if (log(random) < log_acceptance_probability) {
					/* accept the proposal */
					current->items.remove(item_index);
				}
fprintf(stderr, "gibbs_field.sample: done proposing item deletion\n");
			}
		}
	}

private:
	/* NOTE: we assume `neighborhood[0]` refers to the patch at the given `patch_position` */
	template<typename RNGType>
	inline void sample_cell(RNGType& rng,
			patch_type* neighborhood[4],
			unsigned int neighbor_count,
			const position& patch_position,
			const position& world_position)
	{
		/* compute the old item type and index */
		patch_type& current_patch = *neighborhood[0];
		unsigned int old_item_index = 0, old_item_type = cache.item_type_count;
		for (unsigned int m = 0; m < current_patch.items.length; m++) {
			if (current_patch.items[m].location == world_position) {
				old_item_type = current_patch.items[m].item_type;
				old_item_index = m; break;
			}
		}

		float* log_probabilities = (float*) alloca(sizeof(float) * (cache.item_type_count + 1));
		for (unsigned int i = 0; i < cache.item_type_count; i++)
			log_probabilities[i] = cache.intensity(world_position, i);
		for (unsigned int j = 0; j < neighbor_count; j++) {
			const auto& items = neighborhood[j]->items;
			for (unsigned int m = 0; m < items.length; m++) {
				for (unsigned int i = 0; i < cache.item_type_count; i++)
					/* compute the energy contribution of this cell when the item type is `i` */
					log_probabilities[i] += cache.interaction(world_position, items[m].location, i, items[m].item_type);
			}
		}

		log_probabilities[cache.item_type_count] = 0.0;
		normalize_exp(log_probabilities, cache.item_type_count + 1);
		float random = (float) rng() / rng.max();
		unsigned int sampled_item_type = select_categorical(
				log_probabilities, random, cache.item_type_count + 1);

		if (old_item_type == sampled_item_type) {
			/* the Gibbs step didn't change anything */
			return;
		} if (old_item_type < cache.item_type_count) {
			/* remove the old item position */
			current_patch.items.remove(old_item_index);
		} if (sampled_item_type < cache.item_type_count) {
			/* add the new item position */
			current_patch.items.add({sampled_item_type, world_position, 0, 0});
		}
	}
};

} /* namespace nel */

#endif /* NEL_GIBBS_FIELD_H_ */
