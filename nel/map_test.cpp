#include "map.h"

using namespace nel;

struct empty_data {
	static inline void move(const empty_data& src, empty_data& dst) { }
	static inline void free(empty_data& data) { }
};

constexpr bool init(empty_data& data) { return true; }

struct item_position_printer { };

template<typename Stream>
inline bool print(const item& item, Stream& out, const item_position_printer& printer) {
	return print(item.location, out);
}

float intensity(const position& world_position, unsigned int type, float* args) {
	return args[type];
}

float interaction(
		const position& first_position, const position& second_position,
		unsigned int first_type, unsigned int second_type, float* args)
{
	unsigned int item_type_count = (unsigned int) args[0];
	float first_cutoff = args[4 * (first_type * item_type_count + second_type) + 1];
	float second_cutoff = args[4 * (first_type * item_type_count + second_type) + 2];
	float first_value = args[4 * (first_type * item_type_count + second_type) + 3];
	float second_value = args[4 * (first_type * item_type_count + second_type) + 4];

	uint64_t squared_length = (first_position - second_position).squared_length();
	if (squared_length < first_cutoff)
		return first_value;
	else if (squared_length < second_cutoff)
		return second_value;
	else return 0.0;
}

int main(int argc, const char** argv) {
	static constexpr int n = 32;
	float intensity_per_item[] = { -2.0f };
	float interaction_args[] = { 1, 40.0f, 200.0f, 0.0f, -40.0f };
	auto m = map<empty_data>(n, 1, 10, constant_intensity_fn, intensity_per_item, piecewise_box_interaction_fn, interaction_args);

	patch<empty_data>* neighborhood[4];
	position neighbor_positions[4];
	m.get_fixed_neighborhood({0, 0}, neighborhood, neighbor_positions);

	array<item> items(128);
	m.get_items({-2*n, -2*n}, {2*n - 1, 2*n - 1}, items);

	FILE* out = stdout;
	item_position_printer printer;
	print(items, out, printer); print('\n', out); fflush(out);
 	return EXIT_SUCCESS;
}
