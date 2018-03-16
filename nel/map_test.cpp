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
