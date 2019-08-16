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

#ifndef JBW_MAP_H_
#define JBW_MAP_H_

#include <core/map.h>
#include "gibbs_field.h"

namespace jbw {

using namespace core;

struct item {
	unsigned int item_type;

	/* the position of the item, in world coordinates */
	position location;

	/* a time of 0 indicates the item always existed */
	uint64_t creation_time;

	/* a time of 0 indicates the item was never deleted */
	uint64_t deletion_time;

	static inline void move(const item& src, item& dst) {
		dst.item_type = src.item_type;
		dst.location = src.location;
		dst.creation_time = src.creation_time;
		dst.deletion_time = src.deletion_time;
	}
};

template<typename Stream>
bool read(item& i, Stream& in) {
	return read(i.item_type, in)
		&& read(i.location, in)
		&& read(i.creation_time, in)
		&& read(i.deletion_time, in);
}

template<typename Stream>
bool write(const item& i, Stream& out) {
	return write(i.item_type, out)
		&& write(i.location, out)
		&& write(i.creation_time, out)
		&& write(i.deletion_time, out);
}

template<typename Data>
struct patch
{
	array<item> items;

	/**
	 * Indicates if this patch is fixed, or if it can be resampled (for
	 * example, if it's on the edge)
	 */
	bool fixed;

	Data data;

	static inline void move(const patch& src, patch& dst) {
		core::move(src.items, dst.items);
		core::move(src.data, dst.data);
		dst.fixed = src.fixed;
	}

	static inline void free(patch& p) {
		core::free(p.items);
		core::free(p.data);
	}
};

template<typename Data>
inline bool init(patch<Data>& new_patch) {
	new_patch.fixed = false;
	if (!init(new_patch.data)) {
		return false;
	} else if (!array_init(new_patch.items, 8)) {
		fprintf(stderr, "init ERROR: Insufficient memory for patch.items.\n");
		free(new_patch.data); return false;
	}
	return true;
}

template<typename Data>
inline bool init(patch<Data>& new_patch,
		const array<item>& src_items,
		const position item_position_offset)
{
	new_patch.fixed = false;
	if (!init(new_patch.data)) {
		return false;
	} else if (!array_init(new_patch.items, src_items.capacity)) {
		fprintf(stderr, "init ERROR: Insufficient memory for patch.items.\n");
		free(new_patch.data); return false;
	}
	for (unsigned int i = 0; i < src_items.length; i++) {
		new_patch.items[i].item_type = src_items[i].item_type;
		new_patch.items[i].location = src_items[i].location + item_position_offset;
		new_patch.items[i].creation_time = 0;
		new_patch.items[i].deletion_time = 0;
	}
	new_patch.items.length = src_items.length;
	return true;
}

template<typename Data, typename Stream, typename... DataReader>
bool read(patch<Data>& p, Stream& in, DataReader&&... reader) {
	if (!read(p.fixed, in) || !read(p.items, in)) {
		return false;
	} else if (!read(p.data, in, std::forward<DataReader>(reader)...)) {
		free(p.items);
		return false;
	}
	return true;
}

template<typename Data, typename Stream, typename... DataWriter>
bool write(const patch<Data>& p, Stream& out, DataWriter&&... writer) {
	return write(p.fixed, out)
		&& write(p.items, out)
		&& write(p.data, out, std::forward<DataWriter>(writer)...);
}

template<typename K, typename V>
inline unsigned int binary_search(const array_map<K, V>& a, const K& b) {
	if (a.size == 0) return 0;
	return core::binary_search(a.keys, b, 0, a.size - 1);
}

template<typename T>
void shift_right(T* list, unsigned int length,
		unsigned int index, unsigned int shift)
{
	for (unsigned int i = length + shift - 1; i > index + shift - 1; i--)
		move(list[i - shift], list[i]);
}

template<typename K, typename V, typename ApplyFunction>
inline bool apply_contiguous(
		const array_map<K, V>& sorted_map,
		const K& min, unsigned int count,
		ApplyFunction apply)
{
	unsigned int num_matching_elements = 0;
	unsigned int i = binary_search(sorted_map, min);
	for (unsigned int j = 0; j < count; j++) {
		if (i + num_matching_elements < sorted_map.size && sorted_map.keys[i + num_matching_elements] == min + j) {
			if (!apply(sorted_map.values[i + num_matching_elements], min + j)) return false;
			num_matching_elements++;
		}
	}
	return true;
}

template<typename K, typename V, typename InitFunction>
inline unsigned int get_or_init_contiguous(
		array_map<K, V>& sorted_map,
		unsigned int binary_search_index,
		const K& min, uint_fast8_t count,
		InitFunction init)
{
	unsigned int num_matching_elements = 0;
	unsigned int i = binary_search_index;
	bool* matches = (bool*) alloca(sizeof(bool) * count);
	for (uint_fast8_t j = 0; j < count; j++) {
		if (i + num_matching_elements < sorted_map.size && sorted_map.keys[i + num_matching_elements] == min + j) {
			num_matching_elements++;
			matches[j] = true;
		} else {
			matches[j] = false;
		}
	}

	shift_right(sorted_map.keys, sorted_map.size, i + num_matching_elements, count - num_matching_elements);
	shift_right(sorted_map.values, sorted_map.size, i + num_matching_elements, count - num_matching_elements);

	unsigned int remaining_matching_elements = num_matching_elements;
	for (uint_fast8_t j = count; j > 0; j--) {
		if (matches[j - 1]) {
			move(sorted_map.keys[i + remaining_matching_elements - 1], sorted_map.keys[i + j - 1]);
			move(sorted_map.values[i + remaining_matching_elements - 1], sorted_map.values[i + j - 1]);
			remaining_matching_elements--;
		} else {
			sorted_map.keys[i + j - 1] = min + j - 1;
			init(sorted_map.values[i + j - 1], min + j - 1);
		}
	}
	sorted_map.size += count - num_matching_elements;
	return i;
}

template<typename K, typename V, typename InitFunction>
inline unsigned int get_or_init_contiguous(
		array_map<K, V>& sorted_map,
		const K& min, uint_fast8_t count,
		InitFunction init)
{
	return get_or_init_contiguous(sorted_map, binary_search(sorted_map, min), min, count, init);
}

template<typename T>
bool is_sorted_and_distinct(const T* a, size_t length) {
	for (size_t i = 1; i < length; i++) {
		if (a[i] <= a[i - 1]) return false;
	}
	return true;
}

template<typename PerPatchData, typename ItemType>
struct map
{
	array_map<int64_t, array_map<int64_t, patch<PerPatchData>>> patches;

	unsigned int n;
	unsigned int mcmc_iterations;

	std::minstd_rand rng;
	uint_fast32_t initial_seed;
	gibbs_field_cache<ItemType> cache;

	typedef patch<PerPatchData> patch_type;
	typedef ItemType item_type;

public:
	map(unsigned int n, unsigned int mcmc_iterations, const ItemType* item_types, unsigned int item_type_count, uint_fast32_t seed) :
		patches(32), n(n), mcmc_iterations(mcmc_iterations), initial_seed(seed), cache(item_types, item_type_count, n)
	{
		rng.seed(seed);
#if !defined(NDEBUG)
		rng.seed(0);
#else
		rng.seed((uint_fast32_t) milliseconds());
#endif
	}

	map(unsigned int n, unsigned int mcmc_iterations, const ItemType* item_types, unsigned int item_type_count) :
		map(n, mcmc_iterations, item_types, item_type_count,
#if !defined(NDEBUG)
			0) { }
#else
			(uint_fast32_t) milliseconds()) { }
#endif

	~map() { free_helper(); }

	inline patch_type& get_existing_patch(const position& patch_position)
	{
		unsigned int i = binary_search(patches, patch_position.y);
#if !defined(NDEBUG)
		if (i == patches.size || patches.keys[i] > patch_position.y)
			fprintf(stderr, "map.get_existing_patch WARNING: The requested patch does not exist.\n");
#endif

		array_map<int64_t, patch_type>& row = patches.values[i];
		i = binary_search(row, patch_position.x);
#if !defined(NDEBUG)
		if (i == row.size || row.keys[i] > patch_position.x)
			fprintf(stderr, "map.get_existing_patch WARNING: The requested patch does not exist.\n");
#endif

		return row.values[i];
	}

	/**
	 * Returns the patches in the world that intersect with a bounding box of
	 * size n centered at `world_position`. This function will create any
	 * missing patches and ensure that the returned patches are 'fixed': they
	 * cannot be modified by future sampling. The patches and their positions
	 * are returned in row-major order, and the function returns the index in
	 * `neighborhood` of the patch containing `world_position`.
	 */
	unsigned int get_fixed_neighborhood(
			position world_position,
			patch_type* neighborhood[4],
			position out_patch_positions[4])
	{
		unsigned int index = get_neighborhood_positions(world_position, out_patch_positions);

		int64_t min_y = out_patch_positions[2].y;
		int64_t min_x = out_patch_positions[0].x;

		/* first check the four rows exist in `patches`, and if not, create them */
		patches.ensure_capacity(patches.size + 4);

		bool fixed_bottom_left;
		bool fixed_bottom_right;
		bool fixed_top_left;
		bool fixed_top_right;

		unsigned int i = binary_search(patches, min_y);
		unsigned int row_index = i;
		unsigned int column_indices[4];
		if (i < patches.size && patches.keys[i] == min_y) {
			array_map<int64_t, patch_type>& bottom_row = patches.values[i];
			unsigned int j = binary_search(bottom_row, min_x);
			column_indices[1] = j;
			if (j < bottom_row.size && bottom_row.keys[j] == min_x) {
				fixed_bottom_left = bottom_row.values[j].fixed;
				j++;
			} else {
				fixed_bottom_left = false;
			}

			if (j < bottom_row.size && bottom_row.keys[j] == min_x + 1) {
				fixed_bottom_right = bottom_row.values[j].fixed;
			} else {
				fixed_bottom_right = false;
			}
			i++;
		} else {
			column_indices[1] = 0;
			fixed_bottom_left = false;
			fixed_bottom_right = false;
		}

		if (i < patches.size && patches.keys[i] == min_y + 1) {
			array_map<int64_t, patch_type>& top_row = patches.values[i];
			unsigned int j = binary_search(top_row, min_x);
			column_indices[2] = j;
			if (j < top_row.size && top_row.keys[j] == min_x) {
				fixed_top_left = top_row.values[j].fixed;
				j++;
			} else {
				fixed_top_left = false;
			}

			if (j < top_row.size && top_row.keys[j] == min_x + 1) {
				fixed_top_right = top_row.values[j].fixed;
			} else {
				fixed_top_right = false;
			}
		} else {
			column_indices[2] = 0;
			fixed_top_left = false;
			fixed_top_right = false;
		}

	int64_t start_x[4];
	uint_fast8_t column_counts[4];
	if (fixed_bottom_left) {
		if (fixed_bottom_right) {
			if (fixed_top_left) {
				if (fixed_top_right) {
					column_counts[0] = 0;
					column_counts[1] = 0;
					column_counts[2] = 0;
					column_counts[3] = 0;
				} else {
					column_counts[0] = 0;
					column_counts[1] = 3;
					column_counts[2] = 3;
					column_counts[3] = 3;
					start_x[1] = min_x;
					start_x[2] = min_x;
					start_x[3] = min_x;
				}
			} else {
				if (fixed_top_right) {
					column_counts[0] = 0;
					column_counts[1] = 3;
					column_counts[2] = 3;
					column_counts[3] = 3;
					start_x[1] = min_x - 1;
					start_x[2] = min_x - 1;
					start_x[3] = min_x - 1;
				} else {
					column_counts[0] = 0;
					column_counts[1] = 4;
					column_counts[2] = 4;
					column_counts[3] = 4;
					start_x[1] = min_x - 1;
					start_x[2] = min_x - 1;
					start_x[3] = min_x - 1;
				}
			}
		} else {
			if (fixed_top_left) {
				if (fixed_top_right) {
					column_counts[0] = 3;
					column_counts[1] = 3;
					column_counts[2] = 3;
					column_counts[3] = 0;
					start_x[0] = min_x;
					start_x[1] = min_x;
					start_x[2] = min_x;
				} else {
					column_counts[0] = 3;
					column_counts[1] = 3;
					column_counts[2] = 3;
					column_counts[3] = 3;
					start_x[0] = min_x;
					start_x[1] = min_x;
					start_x[2] = min_x;
					start_x[3] = min_x;
				}
			} else {
				if (fixed_top_right) {
					column_counts[0] = 3;
					column_counts[1] = 4;
					column_counts[2] = 4;
					column_counts[3] = 3;
					start_x[0] = min_x;
					start_x[1] = min_x - 1;
					start_x[2] = min_x - 1;
					start_x[3] = min_x - 1;
				} else {
					column_counts[0] = 3;
					column_counts[1] = 4;
					column_counts[2] = 4;
					column_counts[3] = 4;
					start_x[0] = min_x;
					start_x[1] = min_x - 1;
					start_x[2] = min_x - 1;
					start_x[3] = min_x - 1;
				}
			}
		}
	} else {
		if (fixed_bottom_right) {
			if (fixed_top_left) {
				if (fixed_top_right) {
					column_counts[0] = 3;
					column_counts[1] = 3;
					column_counts[2] = 3;
					column_counts[3] = 0;
					start_x[0] = min_x - 1;
					start_x[1] = min_x - 1;
					start_x[2] = min_x - 1;
				} else {
					column_counts[0] = 3;
					column_counts[1] = 4;
					column_counts[2] = 4;
					column_counts[3] = 3;
					start_x[0] = min_x - 1;
					start_x[1] = min_x - 1;
					start_x[2] = min_x - 1;
					start_x[3] = min_x;
				}
			} else {
				if (fixed_top_right) {
					column_counts[0] = 3;
					column_counts[1] = 3;
					column_counts[2] = 3;
					column_counts[3] = 3;
					start_x[0] = min_x - 1;
					start_x[1] = min_x - 1;
					start_x[2] = min_x - 1;
					start_x[3] = min_x - 1;
				} else {
					column_counts[0] = 3;
					column_counts[1] = 4;
					column_counts[2] = 4;
					column_counts[3] = 4;
					start_x[0] = min_x - 1;
					start_x[1] = min_x - 1;
					start_x[2] = min_x - 1;
					start_x[3] = min_x - 1;
				}
			}
		} else {
			if (fixed_top_left) {
				if (fixed_top_right) {
					column_counts[0] = 4;
					column_counts[1] = 4;
					column_counts[2] = 4;
					column_counts[3] = 0;
					start_x[0] = min_x - 1;
					start_x[1] = min_x - 1;
					start_x[2] = min_x - 1;
				} else {
					column_counts[0] = 4;
					column_counts[1] = 4;
					column_counts[2] = 4;
					column_counts[3] = 3;
					start_x[0] = min_x - 1;
					start_x[1] = min_x - 1;
					start_x[2] = min_x - 1;
					start_x[3] = min_x;
				}
			} else {
				if (fixed_top_right) {
					column_counts[0] = 4;
					column_counts[1] = 4;
					column_counts[2] = 4;
					column_counts[3] = 3;
					start_x[0] = min_x - 1;
					start_x[1] = min_x - 1;
					start_x[2] = min_x - 1;
					start_x[3] = min_x - 1;
				} else {
					column_counts[0] = 4;
					column_counts[1] = 4;
					column_counts[2] = 4;
					column_counts[3] = 4;
					start_x[0] = min_x - 1;
					start_x[1] = min_x - 1;
					start_x[2] = min_x - 1;
					start_x[3] = min_x - 1;
				}
			}
		}
	}

		int64_t start_y;
		uint_fast8_t row_count;
		if (column_counts[0] != 0) {
			start_y = min_y - 1;
			row_count = (column_counts[3] == 0) ? 3 : 4;
			if (row_index > 0 && patches.keys[row_index - 1] == min_y - 1) row_index--;
		} else if (column_counts[1] != 0) {
			start_y = min_y;
			row_count = 3;
		} else {
			/* set `neighborhood` and return since all patches are fixed and return */
			neighborhood[0] = &patches.values[row_index + 1].values[column_indices[2]];
			neighborhood[1] = &patches.values[row_index + 1].values[column_indices[2] + 1];
			neighborhood[2] = &patches.values[row_index].values[column_indices[1]];
			neighborhood[3] = &patches.values[row_index].values[column_indices[1] + 1];
			return index;
		}

		bool first = (patches.size == 0);
		row_index = get_or_init_contiguous(patches, row_index, start_y, row_count,
			[](array_map<int64_t, patch_type>& row, int64_t y) { return array_map_init(row, 4); });

		if (first) {
			/* our `init_patch` function assumes the map isn't empty, so if it is, create an empty patch */
			patches.values[row_index].keys[0] = start_x[0];
			init(patches.values[row_index].values[0]);
			patches.values[row_index].size++;
		}

		i = row_index;
		if (column_counts[0] != 0) {
			patches.values[i].ensure_capacity(patches.values[i].size + column_counts[0]);
			column_indices[0] = get_or_init_contiguous(patches.values[i], start_x[0], column_counts[0],
				[&](patch_type& p, int64_t x) {
					return init_patch(p, position(x, min_y - 1));
				});
			i++;
		}
		if (column_indices[1] > 0 && patches.values[i].keys[column_indices[1] - 1] == start_x[1])
			column_indices[1]--;
		patches.values[i].ensure_capacity(patches.values[i].size + column_counts[1]);
		column_indices[1] = get_or_init_contiguous(patches.values[i], column_indices[1], start_x[1], column_counts[1],
			[&](patch_type& p, int64_t x) {
				return init_patch(p, position(x, min_y));
			});
		i++;
		if (column_indices[2] > 0 && patches.values[i].keys[column_indices[2] - 1] == start_x[2])
			column_indices[2]--;
		patches.values[i].ensure_capacity(patches.values[i].size + column_counts[2]);
		column_indices[2] = get_or_init_contiguous(patches.values[i], column_indices[2], start_x[2], column_counts[2],
			[&](patch_type& p, int64_t x) {
				return init_patch(p, position(x, min_y + 1));
			});
		i++;
		if (column_counts[3] != 0) {
			patches.values[i].ensure_capacity(patches.values[i].size + column_counts[3]);
			column_indices[3] = get_or_init_contiguous(patches.values[i], start_x[3], column_counts[3],
				[&](patch_type& p, int64_t x) {
					return init_patch(p, position(x, min_y + 2));
				});
		}

		/* get the neighborhoods of all the patches */
		position patch_positions[16];
		patch_neighborhood<patch_type> neighborhoods[16];
		unsigned int num_patches_to_sample = 0;

		i = row_index;
		for (uint_fast8_t u = 0; u < 4; u++) {
			for (uint_fast8_t v = 0; v < column_counts[u]; v++) {
				if (patches.values[i].values[column_indices[u] + v].fixed) continue;
				position patch_position = position(patches.values[i].keys[column_indices[u] + v], patches.keys[i]);
				patch_positions[num_patches_to_sample] = patch_position;
				get_neighborhood(patch_position, i, column_indices[u] + v, neighborhoods[num_patches_to_sample++]);
			}
			if (column_counts[u] != 0) i++;
		}

		/* construct the Gibbs field and sample the patches at positions_to_sample */
		gibbs_field<map<PerPatchData, ItemType>> field(
				cache, patch_positions, neighborhoods, num_patches_to_sample, n);
		for (unsigned int i = 0; i < mcmc_iterations; i++)
			field.sample(rng);

		/* set the core four patches to fixed */
		i = row_index;
		if (start_y == min_y - 1) i++;
		unsigned int j0 = column_indices[1];
		if (start_x[1] == min_x - 1) j0++;
		unsigned int j1 = column_indices[2];
		if (start_x[2] == min_x - 1) j1++;
		neighborhood[0] = &patches.values[i + 1].values[j1];
		neighborhood[1] = &patches.values[i + 1].values[j1 + 1];
		neighborhood[2] = &patches.values[i].values[j0];
		neighborhood[3] = &patches.values[i].values[j0 + 1];
		for (unsigned int k = 0; k < 4; k++)
			neighborhood[k]->fixed = true;

		return index;
	}

	/**
	 * Returns the patches in the world that intersect with a bounding box of
	 * size n centered at `world_position`. This function will not create any
	 * missing patches or fix any patches. The number intersecting patches is
	 * returned.
	 */
	unsigned int get_neighborhood(
			position world_position,
			patch_type* neighborhood[4],
			position patch_positions[4])
	{
		get_neighborhood_positions(world_position, patch_positions);

		int64_t min_y = patch_positions[2].y;
		int64_t min_x = patch_positions[0].x;

		unsigned int index = 0;
		apply_contiguous(patches, min_y, 2, [&](const array_map<int64_t, patch_type>& row, int64_t y) {
			return apply_contiguous(row, min_x, 2, [&](patch_type& p, int64_t x) {
				neighborhood[index++] = &p;
				return true;
			});
		});

		return index;
	}

	inline void world_to_patch_coordinates(
			position world_position,
			position& patch_position) const
	{
		int64_t x_quotient = floored_div(world_position.x, n);
		int64_t y_quotient = floored_div(world_position.y, n);
		patch_position = {x_quotient, y_quotient};
	}

	inline void world_to_patch_coordinates(
			position world_position,
			position& patch_position,
			position& position_within_patch) const
	{
		lldiv_t x_quotient = floored_div_with_remainder(world_position.x, n);
		lldiv_t y_quotient = floored_div_with_remainder(world_position.y, n);
		patch_position = {x_quotient.quot, y_quotient.quot};
		position_within_patch = {x_quotient.rem, y_quotient.rem};
	}

	static inline void free(map& world) {
		world.free_helper();
		core::free(world.patches);
		core::free(world.cache);
		world.rng.~linear_congruential_engine();
	}

private:
	inline int64_t floored_div(int64_t a, unsigned int b) const {
		lldiv_t result = lldiv(a, b);
		if (a < 0 && result.rem != 0)
			result.quot--;
		return result.quot;
	}

	inline lldiv_t floored_div_with_remainder(int64_t a, unsigned int b) const {
		lldiv_t result = lldiv(a, b);
		if (a < 0 && result.rem != 0) {
			result.quot--;
			result.rem += b;
		}
		return result;
	}

	inline bool init_patch(patch_type& p, const position& patch_position) {
		/* uniformly sample an existing patch to initialize the new patch */
		if (patches.size > 0) {
			/* copy the items from the existing patch into the new patch */
			unsigned int i;
			do {
				i = rng() % patches.size;
			} while (patches.values[i].size == 0);

			const array_map<int64_t, patch_type>& sampled_row = patches.values[i];
			unsigned int j = rng() % sampled_row.size;
			const patch_type& sampled_patch = sampled_row.values[j];
			if (!init(p, sampled_patch.items, (patch_position - position(sampled_row.keys[j], patches.keys[i])) * n))
				return false;
		} else {
			/* there are no patches so initialize an empty patch */
			if (!init(p)) return false;
		}
		return true;
	}

	inline void get_neighborhood(
			position patch_position, unsigned int row_index,
			unsigned int column_index, patch_neighborhood<patch_type>& n)
	{
		int64_t x = patch_position.x;
		int64_t y = patch_position.y;

		const array_map<int64_t, patch_type>& current_row = patches.values[row_index];
		n.bottom_left_neighborhood[0] = &current_row.values[column_index];
		n.bottom_right_neighborhood[0] = &current_row.values[column_index];
		n.top_left_neighborhood[0] = &current_row.values[column_index];
		n.top_right_neighborhood[0] = &current_row.values[column_index];
		n.bottom_left_neighbor_count = 1;
		n.bottom_right_neighbor_count = 1;
		n.top_left_neighbor_count = 1;
		n.top_right_neighbor_count = 1;

		/* check if this row has patches to the right and left of the current patch */
		if (column_index > 0 && current_row.keys[column_index - 1] == x - 1) {
			n.bottom_left_neighborhood[n.bottom_left_neighbor_count++] = &current_row.values[column_index - 1];
			n.top_left_neighborhood[n.top_left_neighbor_count++] = &current_row.values[column_index - 1];
		} if (column_index + 1 < current_row.size && current_row.keys[column_index + 1] == x + 1) {
			n.bottom_right_neighborhood[n.bottom_right_neighbor_count++] = &current_row.values[column_index + 1];
			n.top_right_neighborhood[n.top_right_neighbor_count++] = &current_row.values[column_index + 1];
		}

		/* check if there are rows above and below the `current_row` */
		if (row_index > 0 && patches.keys[row_index - 1] == y - 1) {
			const array_map<int64_t, patch_type>& row = patches.values[row_index - 1];
			unsigned int i = binary_search(row, x - 1);
			if (i < row.size && row.keys[i] == x - 1) {
				n.bottom_left_neighborhood[n.bottom_left_neighbor_count++] = &row.values[i];
				i++;
			} if (i < row.size && row.keys[i] == x) {
				n.bottom_left_neighborhood[n.bottom_left_neighbor_count++] = &row.values[i];
				n.bottom_right_neighborhood[n.bottom_right_neighbor_count++] = &row.values[i];
				i++;
			} if (i < row.size && row.keys[i] == x + 1) {
				n.bottom_right_neighborhood[n.bottom_right_neighbor_count++] = &row.values[i];
			}
		} if (row_index + 1 < patches.size && patches.keys[row_index + 1] == y + 1) {
			const array_map<int64_t, patch_type>& row = patches.values[row_index + 1];
			unsigned int i = binary_search(row, x - 1);
			if (i < row.size && row.keys[i] == x - 1) {
				n.top_left_neighborhood[n.top_left_neighbor_count++] = &row.values[i];
				i++;
			} if (i < row.size && row.keys[i] == x) {
				n.top_left_neighborhood[n.top_left_neighbor_count++] = &row.values[i];
				n.top_right_neighborhood[n.top_right_neighbor_count++] = &row.values[i];
				i++;
			} if (i < row.size && row.keys[i] == x + 1) {
				n.top_right_neighborhood[n.top_right_neighbor_count++] = &row.values[i];
			}
		}
	}

	/**
	 * Retrieves the positions of four patches that contain the bounding box of
	 * size n centered at `world_position`. The positions are stored in
	 * `patch_positions` in row-major order, and the function returns the index
	 * of the patch containing `world_position`.
	 */
	unsigned int get_neighborhood_positions(
			position world_position,
			position patch_positions[4])
	{
		position patch_position, position_within_patch;
		world_to_patch_coordinates(world_position, patch_position, position_within_patch);

		/* determine the quadrant of our current location in current_patch */
		unsigned int patch_index;
		if (position_within_patch.x < (n / 2)) {
			/* we are in the left half of this patch */
			if (position_within_patch.y < (n / 2)) {
				/* we are in the bottom-left quadrant */
				patch_positions[0] = patch_position.left();
				patch_index = 1;
			} else {
				/* we are in the top-left quadrant */
				patch_positions[0] = patch_position.left().up();
				patch_index = 3;
			}
		} else {
			/* we are in the right half of this patch */
			if (position_within_patch.y < (n / 2)) {
				/* we are in the bottom-right quadrant */
				patch_positions[0] = patch_position;
				patch_index = 0;
			} else {
				/* we are in the top-right quadrant */
				patch_positions[0] = patch_position.up();
				patch_index = 2;
			}
		}

		patch_positions[1] = patch_positions[0].right();
		patch_positions[2] = patch_positions[0].down();
		patch_positions[3] = patch_positions[2].right();
		return patch_index;
	}

	inline void free_helper() {
		for (auto row : patches) {
			for (auto entry : row.value)
				core::free(entry.value);
			core::free(row.value);
		}
	}

	bool is_valid() {
		if (!is_sorted_and_distinct(patches.keys, patches.size)) {
			fprintf(stderr, "map.is_valid WARNING: Patch rows are not sorted or distinct.\n");
			return false;
		}
		for (const auto& row : patches) {
			if (!is_sorted_and_distinct(row.value.keys, row.value.size)) {
				fprintf(stderr, "map.is_valid WARNING: Found a row with patches that are not sorted or distinct.\n");
				return false;
			}
		}
		return true;
	}
};

template<typename PerPatchData, typename ItemType>
inline bool init(map<PerPatchData, ItemType>& world, unsigned int n,
		unsigned int mcmc_iterations, const ItemType* item_types,
		unsigned int item_type_count, uint_fast32_t seed)
{
	if (!array_map_init(world.patches, 32))
		return false;
	world.n = n;
	world.mcmc_iterations = mcmc_iterations;
	world.initial_seed = seed;
	if (!init(world.cache, item_types, item_type_count, n)) {
		free(world.patches);
		return false;
	}

	new (&world.rng) std::minstd_rand(seed);
	return true;
}

template<typename PerPatchData, typename ItemType>
inline bool init(map<PerPatchData, ItemType>& world, unsigned int n,
		unsigned int mcmc_iterations, const ItemType* item_types,
		unsigned int item_type_count)
{
#if !defined(NDEBUG)
	uint_fast32_t seed = 0;
#else
	uint_fast32_t seed = (uint_fast32_t) milliseconds();
#endif
	return init(world, n, mcmc_iterations, item_types, item_type_count, seed);
}

template<typename PerPatchData, typename ItemType, typename Stream, typename PatchReader>
bool read(map<PerPatchData, ItemType>& world, Stream& in,
		const ItemType* item_types, unsigned int item_type_count,
		PatchReader& patch_reader = default_scribe())
{
	/* read PRNG state into a char* buffer */
	size_t length;
	if (!read(length, in)) return false;
	char* state = (char*) alloca(sizeof(char) * length);
	if (state == NULL || !read(state, in, (unsigned int) length))
		return false;

	std::stringstream buffer(std::string(state, length));
	buffer >> world.rng;

	size_t row_count;
	if (!read(world.n, in)
	 || !read(world.mcmc_iterations, in)
	 || !read(world.initial_seed, in)
	 || !read(row_count, in)
	 || !array_map_init(world.patches, ((size_t) 1) << (core::log2(row_count == 0 ? 1 : row_count) + 1)))
		return false;

	if (!read(world.patches.keys, in, row_count)) {
		free(world.patches);
		return false;
	}
	for (size_t i = 0; i < row_count; i++) {
		size_t column_count;
		array_map<int64_t, patch<PerPatchData>>& row = world.patches.values[i];
		if (!read(column_count, in)
		 || !array_map_init(row, ((size_t) 1) << (core::log2(column_count == 0 ? 1 : column_count) + 1)))
		{
			for (auto row : world.patches) {
				for (auto entry : row.value)
					free(entry.value);
				free(row.value);
			}
			free(world.patches);
			return false;
		}
		world.patches.size++;

		if (!read(row.keys, in, column_count)) {
			for (auto row : world.patches) {
				for (auto entry : row.value)
					free(entry.value);
				free(row.value);
			}
			free(world.patches);
			return false;
		}
		for (size_t j = 0; j < column_count; j++) {
			if (!read(row.values[j], in, patch_reader)) {
				for (auto row : world.patches) {
					for (auto entry : row.value)
						free(entry.value);
					free(row.value);
				}
				free(world.patches);
				return false;
			}
			row.size++;
		}
	}

	if (!init(world.cache, item_types, item_type_count, world.n)) {
		for (auto row : world.patches) {
			for (auto entry : row.value)
				free(entry.value);
			free(row.value);
		}
		free(world.patches);
		return false;
	}
	return true;
}

/* NOTE: this function assumes the variables in the map are not modified during writing */
template<typename PerPatchData, typename ItemType, typename Stream, typename PatchWriter>
bool write(const map<PerPatchData, ItemType>& world, Stream& out,
		PatchWriter& patch_writer = default_scribe())
{
	/* write the PRNG state into a stringstream buffer */
	std::stringstream buffer;
	buffer << world.rng;
	std::string data = buffer.str();
	if (!write(data.length(), out)
	 || !write(data.c_str(), out, (unsigned int) data.length()))
		return false;

	if (!write(world.n, out)
	 || !write(world.mcmc_iterations, out)
	 || !write(world.initial_seed, out)
	 || !write(world.patches.size, out)
	 || !write(world.patches.keys, out, world.patches.size))
		return false;

	for (size_t i = 0; i < world.patches.size; i++) {
		const array_map<int64_t, patch<PerPatchData>>& row = world.patches.values[i];
		if (!write(row.size, out)
		 || !write(row.keys, out, row.size)
		 || !write(row.values, out, row.size, patch_writer))
			return false;
	}
	return true;
}

} /* namespace jbw */

#endif /* JBW_MAP_H_ */
