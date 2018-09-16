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

struct item_properties {
	intensity_function intensity_fn;
	interaction_function* interaction_fns;

	float* intensity_fn_args;
	unsigned int intensity_fn_arg_count;

	float** interaction_fn_args;
	unsigned int* interaction_fn_arg_counts;
};

int main(int argc, const char** argv) {
	static constexpr int n = 32;
	float intensity_per_item[] = { -2.0f };
	interaction_function interaction_fns[] = { piecewise_box_interaction_fn };
	float interaction_arg[] = { 40.0f, 200.0f, 0.0f, -40.0f };
	float* interaction_args[] = { interaction_arg };
	unsigned int interaction_fn_arg_count = 4;

	item_properties item_type;
	item_type.intensity_fn = constant_intensity_fn;
	item_type.interaction_fns = interaction_fns;
	item_type.intensity_fn_args = intensity_per_item;
	item_type.intensity_fn_arg_count = 1;
	item_type.interaction_fn_args = interaction_args;
	item_type.interaction_fn_arg_counts = &interaction_fn_arg_count;
	auto m = map<empty_data, item_properties>(n, 10, &item_type, 1);

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
