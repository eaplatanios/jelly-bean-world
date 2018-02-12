#ifndef NEL_GIBBS_FIELD_H_
#define NEL_GIBBS_FIELD_H_

#include <core/random.h>
#include <math/log.h>
#include "position.h"

namespace nel {

using namespace core;

typedef float (*intensity_function)(const position&, unsigned int);
typedef float (*interaction_function)(const position&, const position&, unsigned int, unsigned int);

template<typename Map>
class gibbs_field
{
	Map& map;
	position* patch_positions;
	unsigned int patch_count;

	intensity_function intensity;
	interaction_function interaction;

	unsigned int n;
	unsigned int item_type_count;

	typedef typename Map::patch_type patch_type;

public:
	/* NOTE: `patches` and `patch_positions` are used directly, and not copied */
	gibbs_field(Map& map, position* patch_positions, unsigned int patch_count,
			unsigned int n, unsigned int item_type_count,
			intensity_function intensity, interaction_function interaction) :
		map(map), patch_positions(patch_positions), patch_count(patch_count),
		intensity(intensity), interaction(interaction), n(n), item_type_count(item_type_count) { }

	~gibbs_field() { }

	void sample() {
		for (unsigned int i = 0; i < patch_count * n * n; i++)
			sample_cell(patch_positions[sample_uniform(patch_count)], {sample_uniform(n), sample_uniform(n)});
	}

private:
	inline void sample_cell(
			const position& patch_position,
			const position& position_within_patch)
	{
		patch_type* neighborhood[4];
		position neighbor_positions[4];
		position world_position = patch_position * n + position_within_patch;

		unsigned int patch_index;
		unsigned int neighbor_count = map.get_neighborhood(world_position, neighborhood, neighbor_positions, patch_index);

		float* log_probabilities = (float*) malloc(sizeof(float) * (item_type_count + 1));
		unsigned int old_item_index = 0, old_item_type = item_type_count;
		for (unsigned int i = 0; i < item_type_count; i++) {
			/* compute the energy contribution of this cell when the item type is `i` */
			log_probabilities[i] = intensity(world_position, i);
			for (unsigned int j = 0; j < neighbor_count; j++) {
				for (unsigned int k = 0; k < item_type_count; k++) {
					const array<position>& item_positions =neighborhood[j]->item_positions[k];
					for (unsigned int m = 0; m < item_positions.length; m++) {
						if (item_positions[m] == world_position) {
							old_item_index = m;
							old_item_type = k;
							continue; /* ignore the current position */
						}

						log_probabilities[i] += interaction(world_position, item_positions[m], i, k);
					}
				}
			}
		}

		log_probabilities[item_type_count] = 0.0;
		normalize_exp(log_probabilities, item_type_count + 1);
		unsigned int sampled_item_type = sample_categorical(log_probabilities, item_type_count + 1);
		free(log_probabilities);

		patch_type& current_patch = *neighborhood[patch_index];
		if (old_item_type == sampled_item_type) {
			/* the Gibbs step didn't change anything */
			return;
		} if (old_item_type < item_type_count) {
			/* remove the old item position */
			current_patch.item_positions[old_item_type].remove(old_item_index);
		} if (sampled_item_type < item_type_count) {
			/* add the new item position */
			current_patch.item_positions[sampled_item_type].add(world_position);
		}
	}
};

} /* namespace nel */

#endif /* NEL_GIBBS_FIELD_H_ */
