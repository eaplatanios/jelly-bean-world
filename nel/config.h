#ifndef NEL_CONFIG_H_
#define NEL_CONFIG_H_

#include <core/array.h>
#include <core/utility.h>
#include "gibbs_field.h"

namespace nel {

using namespace core;

struct item_properties {
	string name;

	float* scent;
	float* color;

	/* energy function parameters for the Gibbs field */
	float intensity;

	static inline void free(item_properties& properties) {
		core::free(properties.name);
		core::free(properties.scent);
		core::free(properties.color);
	}
};

inline bool init(
		item_properties& properties, const item_properties& src,
		unsigned int scent_dimension, unsigned int color_dimension)
{
	properties.name = src.name;
	properties.scent = (float*) malloc(sizeof(float) * scent_dimension);
	if (properties.scent == NULL) {
		fprintf(stderr, "init ERROR: Insufficient memory for item_properties.scent.\n");
		return false;
	}
	properties.color = (float*) malloc(sizeof(float) * color_dimension);
	if (properties.color == NULL) {
		fprintf(stderr, "init ERROR: Insufficient memory for item_properties.scent.\n");
		free(properties.scent); return false;
	}

	for (unsigned int i = 0; i < scent_dimension; i++)
		properties.scent[i] = src.scent[i];
	for (unsigned int i = 0; i < color_dimension; i++)
		properties.color[i] = src.color[i];
	properties.intensity = src.intensity;
	return true;
}

struct simulator_config {
	/* agent capabilities */
	unsigned int max_steps_per_movement;
	unsigned int scent_dimension;
	unsigned int color_dimension;
	unsigned int vision_range;

	/* world properties */
	unsigned int patch_size;
	unsigned int gibbs_iterations;
	array<item_properties> item_types;

	/* TODO: Need to make these serializable somehow (maybe by contraining them). */
	nel::intensity_function intensity;
	nel::interaction_function interaction;

	simulator_config() : item_types(8) { }

	simulator_config(const simulator_config& src) :
		max_steps_per_movement(src.max_steps_per_movement),
		scent_dimension(src.scent_dimension), color_dimension(src.color_dimension),
		vision_range(src.vision_range), patch_size(src.patch_size),
		gibbs_iterations(src.gibbs_iterations), item_types(src.item_types.length), 
		intensity(src.intensity), interaction(src.interaction)
	{
		for (unsigned int i = 0; i < src.item_types.length; i++)
			init(item_types[i], src.item_types[i], scent_dimension, color_dimension);
		item_types.length = src.item_types.length;
	}

	~simulator_config() {
		for (item_properties& properties : item_types)
			core::free(properties);
	}
};

} /* namespace nel */

#endif /* NEL_CONFIG_H_ */
