#ifndef ENERGY_FUNCTIONS_H_
#define ENERGY_FUNCTIONS_H_

#include "position.h"

namespace nel {

typedef float (*intensity_function)(const position, const float*);
typedef float (*interaction_function)(const position, const position, const float*);

typedef uint64_t intensity_fns_type;
enum class intensity_fns : intensity_fns_type {
	ZERO = 0, CONSTANT
};

typedef uint64_t interaction_fns_type;
enum class interaction_fns : interaction_fns_type {
	ZERO = 0, PIECEWISE_BOX, CROSS
};

float zero_intensity_fn(const position pos, const float* args) {
	return 0.0;
}

float constant_intensity_fn(const position pos, const float* args) {
	return args[0];
}

intensity_function get_intensity_fn(intensity_fns type, const float* args, unsigned int num_args)
{
	switch (type) {
	case intensity_fns::ZERO:
		if (num_args != 0) {
			fprintf(stderr, "get_intensity_fn ERROR: A zero intensity function requires zero arguments.");
			return NULL;
		}
		return zero_intensity_fn;
	case intensity_fns::CONSTANT:
		if (num_args == 0) {
			fprintf(stderr, "get_intensity_fn ERROR: A constant intensity function requires an argument.");
			return NULL;
		}
		return constant_intensity_fn;
	}
	fprintf(stderr, "get_intensity_fn ERROR: Unknown intensity function type.");
	return NULL;
}

intensity_fns get_intensity_fn(intensity_function function) {
	if (function == zero_intensity_fn) {
		return intensity_fns::ZERO;
	} else if (function == constant_intensity_fn) {
		return intensity_fns::CONSTANT;
	} else {
		fprintf(stderr, "get_intensity_fn ERROR: Unknown intensity_function.");
		exit(EXIT_FAILURE);
	}
}

float zero_interaction_fn(const position pos1, const position pos2, const float* args) {
	return 0.0;
}

float piecewise_box_interaction_fn(const position pos1, const position pos2, const float* args)
{
	float first_cutoff = args[0];
	float second_cutoff = args[1];
	float first_value = args[2];
	float second_value = args[3];

	uint64_t squared_length = (pos1 - pos2).squared_length();
	if (squared_length < first_cutoff)
		return first_value;
	else if (squared_length < second_cutoff)
		return second_value;
	else return 0.0f;
}

float cross_interaction_fn(const position pos1, const position pos2, const float* args)
{
	const position diff = pos1 - pos2;
	uint64_t dist = max(abs(diff.x), abs(diff.y));
	if (dist <= args[0]) {
		if (diff.x == 0 || diff.y == 0)
			return args[2];
		else return args[4];
	} else if (dist <= args[1]) {
		if (diff.x == 0 || diff.y == 0)
			return args[3];
		else return args[5];
	} else {
		return 0.0f;
	}
}

interaction_function get_interaction_fn(interaction_fns type, const float* args, unsigned int num_args)
{
	switch (type) {
	case interaction_fns::ZERO:
		if (num_args != 0) {
			fprintf(stderr, "get_interaction_fn ERROR: A zero interaction function requires zero arguments.");
			return NULL;
		}
		return zero_interaction_fn;
	case interaction_fns::PIECEWISE_BOX:
		if (num_args != 4) {
			fprintf(stderr, "get_interaction_fn ERROR: A piecewise-box integration function requires 4 arguments.");
			return NULL;
		}
		return piecewise_box_interaction_fn;
	case interaction_fns::CROSS:
		if (num_args != 6) {
			fprintf(stderr, "get_interaction_fn ERROR: A cross integration function requires 6 arguments.");
			return NULL;
		}
		return cross_interaction_fn;
	}
	fprintf(stderr, "get_interaction_fn ERROR: Unknown interaction function type.");
	return NULL;
}

interaction_fns get_interaction_fn(interaction_function function) {
	if (function == zero_interaction_fn) {
		return interaction_fns::ZERO;
	} else if (function == piecewise_box_interaction_fn) {
		return interaction_fns::PIECEWISE_BOX;
	} else if (function == cross_interaction_fn) {
		return interaction_fns::CROSS;
	} else {
		fprintf(stderr, "get_interaction_fn ERROR: Unknown interaction_function.");
		exit(EXIT_FAILURE);
	}
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
	case interaction_fns::CROSS:         function = cross_interaction_fn; return true;
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
	} else if (function == cross_interaction_fn) {
		return write((interaction_fns_type) interaction_fns::CROSS, out);
	} else {
		fprintf(stderr, "write ERROR: Unrecognized interaction function.\n");
		return false;
	}
}

inline bool is_constant(const interaction_function function) {
	return function == zero_interaction_fn;
}

/* NOTE: stationary intensity functions are also constant */
inline bool is_stationary(const intensity_function function) {
	return (function == zero_intensity_fn
		 || function == constant_intensity_fn);
}

inline bool is_stationary(const interaction_function function) {
	return (function == zero_interaction_fn
		 || function == piecewise_box_interaction_fn
		 || function == cross_interaction_fn);
}

} /* namespace nel */

#endif /* ENERGY_FUNCTIONS_H_ */
