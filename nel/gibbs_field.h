#ifndef NEL_GIBBS_FIELD_H_
#define NEL_GIBBS_FIELD_H_

#include <core/random.h>
#include <math/log.h>
#include "position.h"

namespace nel {

using namespace core;

typedef float (*intensity_function)(const position&, unsigned int, float*);

enum class intensity_fns {
	ZERO = 0, CONSTANT = 1
};

intensity_function get_intensity_fn(intensity_fns type, float* args, unsigned int num_args) {
	switch (type) {
		case ZERO: return zero_intensity_fn;
		case CONSTANT:
			if (num_args < 1) {
				fprintf(stderr, "A constant intensity function requires an argument.");
        exit(EXIT_FAILURE);
			}
			return constant_intensity_fn;
		default: 
			fprintf(stderr, "Unknown intensity function type.");
      exit(EXIT_FAILURE);
	}
}

float zero_intensity_fn(const position& pos, unsigned int item_type, float* args) {
	return 0.0;
}

float constant_intensity_fn(const position& pos, unsigned int item_type, float* args) {
	return args[0];
}

typedef float (*interaction_function)(const position&, const position&, unsigned int, unsigned int, float*);

enum class interaction_fns {
	ZERO = 0, PIECEWISE_BOX = 1
};

interaction_function get_interaction_fn(interaction_fns type, float* args, unsigned int num_args) {
	switch (type) {
		case ZERO: return zero_interaction_fn;
		case PIECEWISE_BOX:
			if (num_args < 4) {
				fprintf(stderr, "A piecewise-box integration function requires 4 arguments.");
        exit(EXIT_FAILURE);
			}
			return piecewise_box_interaction_fn;
		default: 
			fprintf(stderr, "Unknown intensity function type.");
      exit(EXIT_FAILURE);
	}
}

float zero_interaction_fn(
		const position& pos1, const position& pos2, unsigned int item_type1, unsigned int item_type2, float* args) {
	return 0.0;
}

float piecewise_box_interaction_fn(
		const position& pos1, const position& pos2, unsigned int item_type1, unsigned int item_type2, float* args) {
	uint64_t squared_length = (pos1 - pos2).squared_length();
	if (squared_length < args[0])
		return args[1];
	else if (squared_length < args[2])
		return args[3];
	else return args[1];
}

template<typename Map>
class gibbs_field
{
	Map& map;
	position* patch_positions;
	unsigned int patch_count;

	unsigned int n;
	unsigned int item_type_count;

	typedef typename Map::patch_type patch_type;

public:
	/* NOTE: `patches` and `patch_positions` are used directly, and not copied */
	gibbs_field(Map& map, position* patch_positions, unsigned int patch_count,
			unsigned int n, unsigned int item_type_count,
			intensity_function intensity, interaction_function interaction) :
		map(map), patch_positions(patch_positions), patch_count(patch_count),
		n(n), item_type_count(item_type_count) { }

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
			log_probabilities[i] = map.intensity(world_position, i);
			for (unsigned int j = 0; j < neighbor_count; j++) {
				for (unsigned int k = 0; k < item_type_count; k++) {
					const array<position>& item_positions = neighborhood[j]->item_positions[k];
					for (unsigned int m = 0; m < item_positions.length; m++) {
						if (item_positions[m] == world_position) {
							old_item_index = m;
							old_item_type = k;
							continue; /* ignore the current position */
						}

						log_probabilities[i] += map.interaction(world_position, item_positions[m], i, k);
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
