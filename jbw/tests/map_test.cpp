/**
 * Copyright 2019, The Jelly Bean World Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not
 * use this file except in compliance with the License. You may obtain a copy of
 * the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

#include <jbw/map.h>

using namespace jbw;

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

template<typename FunctionType>
struct energy_function {
	FunctionType fn;
	float* args;
	unsigned int arg_count;

	static inline void free(energy_function<FunctionType>& info) {
		core::free(info.args);
	}
};

struct item_properties {
	energy_function<intensity_function> intensity_fn;
	energy_function<interaction_function>* interaction_fns;
};

inline void set_interaction_args(
		item_properties* item_types, unsigned int first_item_type,
		unsigned int second_item_type, interaction_function interaction,
		std::initializer_list<float> args)
{
	item_types[first_item_type].interaction_fns[second_item_type].fn = interaction;
	item_types[first_item_type].interaction_fns[second_item_type].arg_count = (unsigned int) args.size();
	item_types[first_item_type].interaction_fns[second_item_type].args = (float*) malloc(max((size_t) 1, sizeof(float) * args.size()));

	unsigned int counter = 0;
	for (auto i = args.begin(); i != args.end(); i++)
		item_types[first_item_type].interaction_fns[second_item_type].args[counter++] = *i;
}

template<typename PerPatchData, typename ItemType>
void generate_map(
		map<PerPatchData, ItemType>& world,
		const position& bottom_left_corner,
		const position& top_right_corner)
{
	/* make sure enough of the world is generated */
	patch<empty_data>* neighborhood[4]; position patch_positions[4];
	for (int64_t x = bottom_left_corner.x; x <= top_right_corner.x; x += world.n) {
		for (int64_t y = bottom_left_corner.y; y <= top_right_corner.y; y += world.n)
			world.get_fixed_neighborhood(position(x, y), neighborhood, patch_positions);
		world.get_fixed_neighborhood(position(x, top_right_corner.y), neighborhood, patch_positions);
	}
	for (int64_t y = bottom_left_corner.y; y <= top_right_corner.y; y += world.n)
		world.get_fixed_neighborhood(position(top_right_corner.x, y), neighborhood, patch_positions);
	world.get_fixed_neighborhood(top_right_corner, neighborhood, patch_positions);
}

int main(int argc, const char** argv) {
	static constexpr int n = 32;
	static constexpr unsigned int item_type_count = 4;
	static constexpr unsigned int mcmc_iterations = 4000;
	item_properties item_types[item_type_count];

	item_types[0].intensity_fn.fn = constant_intensity_fn;
	item_types[0].intensity_fn.arg_count = 1;
	item_types[0].intensity_fn.args = (float*) malloc(sizeof(float) * 1);
	item_types[0].intensity_fn.args[0] = -5.3f;
	item_types[0].interaction_fns = (energy_function<interaction_function>*)
			malloc(sizeof(energy_function<interaction_function>) * item_type_count);
	item_types[1].intensity_fn.fn = constant_intensity_fn;
	item_types[1].intensity_fn.arg_count = 1;
	item_types[1].intensity_fn.args = (float*) malloc(sizeof(float) * 1);
	item_types[1].intensity_fn.args[0] = -5.0f;
	item_types[1].interaction_fns = (energy_function<interaction_function>*)
			malloc(sizeof(energy_function<interaction_function>) * item_type_count);
	item_types[2].intensity_fn.fn = constant_intensity_fn;
	item_types[2].intensity_fn.arg_count = 1;
	item_types[2].intensity_fn.args = (float*) malloc(sizeof(float) * 1);
	item_types[2].intensity_fn.args[0] = -5.3f;
	item_types[2].interaction_fns = (energy_function<interaction_function>*)
			malloc(sizeof(energy_function<interaction_function>) * item_type_count);
	item_types[3].intensity_fn.fn = constant_intensity_fn;
	item_types[3].intensity_fn.arg_count = 1;
	item_types[3].intensity_fn.args = (float*) malloc(sizeof(float) * 1);
	item_types[3].intensity_fn.args[0] = 0.0f;
	item_types[3].interaction_fns = (energy_function<interaction_function>*)
			malloc(sizeof(energy_function<interaction_function>) * item_type_count);

	set_interaction_args(item_types, 0, 0, piecewise_box_interaction_fn, {10.0f, 200.0f, 0.0f, -6.0f});
	set_interaction_args(item_types, 0, 1, piecewise_box_interaction_fn, {200.0f, 0.0f, -6.0f, -6.0f});
	set_interaction_args(item_types, 0, 2, piecewise_box_interaction_fn, {10.0f, 200.0f, 2.0f, -100.0f});
	set_interaction_args(item_types, 0, 3, zero_interaction_fn, {});
	set_interaction_args(item_types, 1, 0, piecewise_box_interaction_fn, {200.0f, 0.0f, -6.0f, -6.0f});
	set_interaction_args(item_types, 1, 1, zero_interaction_fn, {});
	set_interaction_args(item_types, 1, 2, piecewise_box_interaction_fn, {200.0f, 0.0f, -100.0f, -100.0f});
	set_interaction_args(item_types, 1, 3, zero_interaction_fn, {});
	set_interaction_args(item_types, 2, 0, piecewise_box_interaction_fn, {10.0f, 200.0f, 2.0f, -100.0f});
	set_interaction_args(item_types, 2, 1, piecewise_box_interaction_fn, {200.0f, 0.0f, -100.0f, -100.0f});
	set_interaction_args(item_types, 2, 2, piecewise_box_interaction_fn, {10.0f, 200.0f, 0.0f, -6.0f});
	set_interaction_args(item_types, 2, 3, zero_interaction_fn, {});
	set_interaction_args(item_types, 3, 0, zero_interaction_fn, {});
	set_interaction_args(item_types, 3, 1, zero_interaction_fn, {});
	set_interaction_args(item_types, 3, 2, zero_interaction_fn, {});
	set_interaction_args(item_types, 3, 3, cross_interaction_fn, {10.0f, 15.0f, 20.0f, -200.0f, -20.0f, 1.0f});

	auto m = map<empty_data, item_properties>(n, mcmc_iterations, item_types, item_type_count);

	position bottom_left_corner = {-100, -15};
	position top_right_corner = {100, 15};
	generate_map(m, bottom_left_corner, top_right_corner);

	printf("%" PRId64 ", %" PRId64 "\n", bottom_left_corner.x, bottom_left_corner.y);
	printf("%" PRId64 ", %" PRId64 "\n", top_right_corner.x, top_right_corner.y);

	position bottom_left_patch_position, top_right_patch_position;
	m.world_to_patch_coordinates(bottom_left_corner, bottom_left_patch_position);
	m.world_to_patch_coordinates(top_right_corner, top_right_patch_position);
	apply_contiguous(m.patches, bottom_left_patch_position.y,
		top_right_patch_position.y - bottom_left_patch_position.y + 1,
		[&](const array_map<int64_t, patch<empty_data>>& row, int64_t y)
	{
		return apply_contiguous(row, bottom_left_patch_position.x,
			top_right_patch_position.x - bottom_left_patch_position.x + 1,
			[&](const patch<empty_data>& patch, int64_t x)
		{
			for (const item& i : patch.items) {
				if (i.location.x >= bottom_left_corner.x && i.location.x <= top_right_corner.x
				 && i.location.y >= bottom_left_corner.y && i.location.y <= top_right_corner.y)
				{
					printf("%u, %" PRId64 ", %" PRId64 "\n", i.item_type, i.location.x, i.location.y);
				}
			}
			return true;
		});
	});

	fflush(stdout);
	return EXIT_SUCCESS;
}
