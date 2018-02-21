#ifndef ENERGY_FUNCTIONS_H_
#define ENERGY_FUNCTIONS_H_

#include "position.h"

namespace nel {

typedef float (*intensity_function)(const position&, unsigned int, float*);
typedef float (*interaction_function)(const position&, const position&, unsigned int, unsigned int, float*);

typedef uint8_t intensity_fns_type;
enum class intensity_fns : intensity_fns_type {
	ZERO = 0, CONSTANT = 1
};

typedef uint8_t interaction_fns_type;
enum class interaction_fns : interaction_fns_type {
	ZERO = 0, PIECEWISE_BOX = 1
};

float zero_intensity_fn(const position& pos, unsigned int item_type, float* args) {
	return 0.0;
}

float constant_intensity_fn(const position& pos, unsigned int item_type, float* args) {
	return args[item_type];
}

intensity_function get_intensity_fn(intensity_fns type, float* args,
		unsigned int num_args, unsigned int item_type_count)
{
	switch (type) {
	case intensity_fns::ZERO:
		return zero_intensity_fn;
	case intensity_fns::CONSTANT:
		if (num_args < item_type_count) {
			fprintf(stderr, "get_intensity_fn ERROR: A constant intensity function requires an argument.");
			return NULL;
		}
		return constant_intensity_fn;
	}
	fprintf(stderr, "get_intensity_fn ERROR: Unknown intensity function type.");
	return NULL;
}

float zero_interaction_fn(
		const position& pos1, const position& pos2, unsigned int item_type1, unsigned int item_type2, float* args) {
	return 0.0;
}

float piecewise_box_interaction_fn(
		const position& pos1, const position& pos2,
		unsigned int item_type1, unsigned int item_type2, float* args)
{
	unsigned int item_type_count = (unsigned int) args[0];
	float first_cutoff = args[4 * (item_type1 * item_type_count + item_type2) + 1];
	float second_cutoff = args[4 * (item_type1 * item_type_count + item_type2) + 2];
	float first_value = args[4 * (item_type1 * item_type_count + item_type2) + 3];
	float second_value = args[4 * (item_type1 * item_type_count + item_type2) + 4];

	uint64_t squared_length = (pos1 - pos2).squared_length();
	if (squared_length < first_cutoff)
		return first_value;
	else if (squared_length < second_cutoff)
		return second_value;
	else return 0.0;
}

interaction_function get_interaction_fn(interaction_fns type, float* args,
		unsigned int num_args, unsigned int item_type_count)
{
	switch (type) {
	case interaction_fns::ZERO:
		return zero_interaction_fn;
	case interaction_fns::PIECEWISE_BOX:
		if (num_args < 4 * item_type_count * item_type_count + 1) {
			fprintf(stderr, "get_interaction_fn ERROR: A piecewise-box integration function requires 4 arguments.");
			return NULL;
		}
		return piecewise_box_interaction_fn;
	}
	fprintf(stderr, "get_interaction_fn ERROR: Unknown intensity function type.");
	return NULL;
}

template<typename Stream>
inline bool read(intensity_function& function, Stream& in) {
	intensity_fns_type c;
	if (!read(c, in)) return false;
	switch ((intensity_fns) c) {
	case intensity_fns::ZERO:     function = zero_intensity_fn; return true;
	case intensity_fns::CONSTANT: function = constant_intensity_fn; return true;
	}
	fprintf(stderr, "read ERROR: Unrecognized intensity function.\n");
	return false;
}

template<typename Stream>
inline bool write(const intensity_function& function, Stream& out) {
	if (function == zero_intensity_fn) {
		return write((intensity_fns_type) intensity_fns::ZERO, out);
	} else if (function == constant_intensity_fn) {
		return write((intensity_fns_type) intensity_fns::CONSTANT, out);
	} else {
		fprintf(stderr, "write ERROR: Unrecognized intensity function.\n");
		return false;
	}
}

template<typename Stream>
inline bool read(interaction_function& function, Stream& in) {
	interaction_fns_type c;
	if (!read(c, in)) return false;
	switch ((interaction_fns) c) {
	case interaction_fns::ZERO:          function = zero_interaction_fn; return true;
	case interaction_fns::PIECEWISE_BOX: function = piecewise_box_interaction_fn; return true;
	}
	fprintf(stderr, "read ERROR: Unrecognized interaction function.\n");
	return false;
}

template<typename Stream>
inline bool write(const interaction_function& function, Stream& out) {
	if (function == zero_interaction_fn) {
		return write((interaction_fns_type) interaction_fns::ZERO, out);
	} else if (function == piecewise_box_interaction_fn) {
		return write((interaction_fns_type) interaction_fns::PIECEWISE_BOX, out);
	} else {
		fprintf(stderr, "write ERROR: Unrecognized interaction function.\n");
		return false;
	}
}

} /* namespace nel */

#endif /* ENERGY_FUNCTIONS_H_ */
