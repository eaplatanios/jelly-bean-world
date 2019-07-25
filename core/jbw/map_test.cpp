// Copyright 2019, The Jelly Bean World Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy of
// the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

#include "map.h"

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

int main(int argc, const char** argv) {

	static constexpr int n = 32;
	float intensity_per_item[] = { -5.0f };
	float interaction_arg[] = { 40.0f, 200.0f, 0.0f, -40.0f };

	energy_function<interaction_function> interaction_fn;
	interaction_fn.fn = piecewise_box_interaction_fn;
	interaction_fn.args = interaction_arg;
	interaction_fn.arg_count = 4;

	item_properties item_type;
	item_type.intensity_fn.fn = constant_intensity_fn;
	item_type.intensity_fn.args = intensity_per_item;
	item_type.intensity_fn.arg_count = 1;
	item_type.interaction_fns = &interaction_fn;

	double density_sum = 0.0;
	double density_sum_squared = 0.0;
	for (unsigned int t = 0; true; t++) {
		auto m = map<empty_data, item_properties>(n, 100, &item_type, 1);

		patch<empty_data>* neighborhood[4];
		position neighbor_positions[4];
		m.get_fixed_neighborhood({0, 0}, neighborhood, neighbor_positions);

		array<item> items(128);
		m.get_items({-2*n, -2*n}, {2*n - 1, 2*n - 1}, items);

		FILE* out = stdout;
		//item_position_printer printer;
		//print(items, out, printer); print('\n', out); fflush(out);
		double estimated_density = (double) items.length / (4*4*n*n);
		density_sum += estimated_density;
		density_sum_squared += estimated_density * estimated_density;
		double estimated_mean = density_sum/(t+1);
		printf("[sample %u] average item density = %.10lf, stddev = %.10lf\n", t, estimated_mean, sqrt(density_sum_squared/(t+1) - estimated_mean*estimated_mean));
	}
	return EXIT_SUCCESS;
}
