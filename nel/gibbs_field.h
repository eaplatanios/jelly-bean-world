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
struct gibbs_field_cache
{
	float* intensities;
	float* interactions;
	unsigned int two_n, four_n;
	unsigned int item_type_count;

	template<typename Map>
	gibbs_field_cache(const Map& map, unsigned int n, unsigned int item_type_count,
			bool is_intensity_stationary, bool is_interaction_stationary) :
		intensities(NULL), interactions(NULL),
		two_n(2*n), four_n(4*n), item_type_count(item_type_count)
	{
		if (!init_helper(map, is_intensity_stationary, is_interaction_stationary))
			exit(EXIT_FAILURE);
	}

	~gibbs_field_cache() { free_helper(); }

	template<typename Map, bool Stationary>
	inline float intensity(const Map& map, const position& pos, unsigned int item_type) {
		if (Stationary)
			return intensities[item_type];
		else return map.intensity(pos, item_type);
	}

	template<typename Map, bool Stationary>
	inline float interaction(const Map& map,
			const position& first_position, const position& second_position,
			unsigned int first_item_type, unsigned int second_item_type)
	{
		if (!Stationary) {
			if (first_position == second_position) return 0.0f;
			return map.interaction(first_position, second_position, first_item_type, second_item_type);
		}

		position diff = first_position - second_position + position(two_n, two_n);
#if !defined(NDEBUG)
		if (diff.x < 0 || diff.x >= four_n || diff.y < 0 || diff.y >= four_n) {
			fprintf(stderr, "gibbs_field_cache.interaction WARNING: The "
					"given two positions further than 4*n from each other.");
			return 0.0f;
		}
#endif
		return interactions[((diff.x*four_n + diff.y)*item_type_count + first_item_type)*item_type_count + second_item_type];
	}

	static inline void free(gibbs_field_cache& cache) { cache.free_helper(); }

private:
	template<typename Map>
	inline bool init_helper(const Map& map, bool is_intensity_stationary, bool is_interaction_stationary) {
		if (is_intensity_stationary) {
			intensities = (float*) malloc(sizeof(float) * item_type_count);
			if (intensities == NULL) {
				fprintf(stderr, "gibbs_field_cache.init_helper ERROR: Insufficient memory for intensities.\n");
				return false;
			}

			/* precompute all intensities */
			for (unsigned int i = 0; i < item_type_count; i++)
				intensities[i] = map.intensity(position(0, 0), i);
		}
		if (is_interaction_stationary) {
			interactions = (float*) malloc(sizeof(float) * four_n * four_n * item_type_count * item_type_count);
			if (interactions == NULL) {
				fprintf(stderr, "gibbs_field_cache.init_helper ERROR: Insufficient memory for interactions.\n");
				if (intensities != NULL) core::free(intensities);
				return false;
			}

			/* precompute all interactions */
			for (unsigned int x = 0; x < four_n; x++) {
				for (unsigned int y = 0; y < four_n; y++) {
					for (unsigned int i = 0; i < item_type_count; i++) {
						for (unsigned int j = 0; j < item_type_count; j++) {
							float interaction;
							if (x == two_n && y == two_n)
								interaction = 0.0f;
							else interaction = map.interaction(position(two_n, two_n), position(x, y), i, j);
							interactions[((x*four_n + y)*item_type_count + i)*item_type_count + j] = interaction;
						}
					}
				}
			}
		}
		return true;
	}

	inline void free_helper() {
		if (intensities != NULL)
			core::free(intensities);
		if (interactions != NULL)
			core::free(interactions);
	}

	template<typename A>
	friend bool init(gibbs_field_cache&, const A&,
			unsigned int, unsigned int, bool, bool);
};

template<typename Map>
bool init(gibbs_field_cache& cache, const Map& map,
		unsigned int n, unsigned int item_type_count,
		bool is_intensity_stationary, bool is_interaction_stationary)
{
	cache.intensities = NULL;
	cache.interactions = NULL;
	cache.two_n = 2*n;
	cache.four_n = 4*n;
	cache.item_type_count = item_type_count;
	return cache.init_helper(map, is_intensity_stationary, is_interaction_stationary);
}

template<typename Map, bool StationaryIntensity, bool StationaryInteraction>
class gibbs_field
{
	Map& map;
	gibbs_field_cache& cache;
	position* patch_positions;
	unsigned int patch_count;

	unsigned int n;
	unsigned int item_type_count;

	typedef typename Map::patch_type patch_type;
	typedef typename Map::item_type item_type;

public:
	/* NOTE: `patch_positions` is used directly, and not copied, so the caller maintains ownership */
	gibbs_field(Map& map, gibbs_field_cache& cache, position* patch_positions,
			unsigned int patch_count, unsigned int n, unsigned int item_type_count) :
		map(map), cache(cache), patch_positions(patch_positions), patch_count(patch_count),
		n(n), item_type_count(item_type_count) { }

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
	template<typename RNGType>
	inline unsigned int sample_uniform(RNGType& rng, unsigned int n) {
		return rng() % n;
	}

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
		unsigned int old_item_index = 0, old_item_type = item_type_count;
		for (unsigned int m = 0; m < current_patch.items.length; m++) {
			if (current_patch.items[m].location == world_position) {
				old_item_type = current_patch.items[m].item_type;
				old_item_index = m; break;
			}
		}

		float* log_probabilities = (float*) alloca(sizeof(float) * (item_type_count + 1));
		for (unsigned int i = 0; i < item_type_count; i++)
			log_probabilities[i] = cache.intensity<Map, StationaryIntensity>(map, world_position, i);
		for (unsigned int j = 0; j < neighbor_count; j++) {
			const array<item_type>& items = neighborhood[j]->items;
			for (unsigned int m = 0; m < items.length; m++) {
				for (unsigned int i = 0; i < item_type_count; i++)
					/* compute the energy contribution of this cell when the item type is `i` */
					log_probabilities[i] += cache.interaction<Map, StationaryInteraction>(map, world_position, items[m].location, i, items[m].item_type);
			}
		}

		log_probabilities[item_type_count] = 0.0;
		normalize_exp(log_probabilities, item_type_count + 1);
		float random = (float) rng() / engine.max();
		unsigned int sampled_item_type = select_categorical(
				log_probabilities, random, item_type_count + 1);

		if (old_item_type == sampled_item_type) {
			/* the Gibbs step didn't change anything */
			return;
		} if (old_item_type < item_type_count) {
			/* remove the old item position */
			current_patch.items.remove(old_item_index);
		} if (sampled_item_type < item_type_count) {
			/* add the new item position */
			current_patch.items.add({sampled_item_type, world_position, 0, 0});
		}
	}
};

} /* namespace nel */

#endif /* NEL_GIBBS_FIELD_H_ */
