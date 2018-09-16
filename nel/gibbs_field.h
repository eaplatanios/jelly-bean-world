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
		intensities(NULL), interactions(NULL), two_n(2*n), four_n(4*n),
		item_types(item_types), item_type_count(item_type_count)
	{
		if (!init_helper()) exit(EXIT_FAILURE);
	}

	~gibbs_field_cache() { free_helper(); }

	inline float intensity(const position& pos, unsigned int item_type) {
		if (is_stationary(item_types[item_type].intensity_fn))
			return intensities[item_type];
		else return item_types[item_type].intensity_fn(pos, item_types[item_type].intensity_fn_args);
	}

	inline float interaction(
			const position& first_position, const position& second_position,
			unsigned int first_item_type, unsigned int second_item_type)
	{
		interaction_function interaction = item_types[first_item_type].interaction_fns[second_item_type];
		if (is_constant(interaction) || !is_stationary(interaction)) {
			if (first_position == second_position) return 0.0f;
			return interaction(first_position, second_position, item_types[first_item_type].interaction_fn_args[second_item_type]);
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
			if (is_stationary(item_types[i].intensity_fn))
				intensities[i] = item_types[i].intensity_fn(position(0, 0), item_types[i].intensity_fn_args);

			for (unsigned int j = 0; j < item_type_count; j++) {
				interaction_function interaction = item_types[i].interaction_fns[j];
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
							else value = interaction(position(two_n, two_n), position(x, y), item_types[i].interaction_fn_args[j]);
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
			position patch_position_offset = patch_positions[i] * n;
			auto process_neighborhood = [&](unsigned int x, unsigned int y,
					patch_type* neighborhood[4], unsigned int neighbor_count)
			{
				position world_position = patch_position_offset + position(x, y);
				sample_cell(rng, neighborhood, neighbor_count, patch_positions[i], world_position);
			};

			map.iterate_neighborhoods(patch_positions[i], process_neighborhood);
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
