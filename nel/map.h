#ifndef NEL_MAP_H_
#define NEL_MAP_H_

#include <core/map.h>
#include "gibbs_field.h"

namespace nel {

using namespace core;

struct item_position {
	unsigned int item_type;
	position location;
};

struct patch
{
	/**
	 * For each type of item, we keep an array of the position of each item
	 * instance in this patch, in world coordinates.
	 */
	array<position>* item_positions;

	/**
	 * Indicates if this patch is fixed, or if it can be resampled (for
	 * example, if it's on the edge)
	 */
	bool fixed;

	static inline void move(const patch& src, patch& dst) {
		dst.item_positions = src.item_positions;
		dst.fixed = src.fixed;
	}

	static inline void free(patch& p, unsigned int item_type_count) {
		for (unsigned int i = 0; i < item_type_count; i++)
			core::free(p.item_positions[i]);
		core::free(p.item_positions);
	}
};

inline bool init(patch& new_patch, unsigned int item_type_count) {
	new_patch.fixed = false;
	new_patch.item_positions = (array<position>*) malloc(sizeof(array<position>) * item_type_count);
	if (new_patch.item_positions == NULL) {
		fprintf(stderr, "init ERROR: Insufficient memory for patch.item_positions.\n");
		return false;
	}
	for (unsigned int i = 0; i < item_type_count; i++) {
		if (!array_init(new_patch.item_positions[i], 8)) {
			for (unsigned int j = 0; j < i; j++) free(new_patch.item_positions[j]);
			free(new_patch.item_positions); return false;
		}
	}
	return true;
}

void* alloc_position_keys(size_t n, size_t element_size) {
	position* keys = (position*) malloc(sizeof(position) * n);
	if (keys == NULL) return NULL;
	for (unsigned int i = 0; i < n; i++)
		position::set_empty(keys[i]);
	return (void*) keys;
}

struct map {
	hash_map<position, patch> patches;

	intensity_function intensity_fn;
	interaction_function interaction_fn;

	// We assume that the length of the args arrays is known at this point and has been checked.
	float* intensity_fn_args;
	float* interaction_fn_args;

	unsigned int n;
	unsigned int item_type_count;
	unsigned int gibbs_iterations;

	typedef patch patch_type;

public:
	map(unsigned int n, unsigned int item_type_count, unsigned int gibbs_iterations,
			intensity_function intensity_fn, float* intensity_fn_args, 
			interaction_function interaction_fn, float* interaction_fn_args) :
		patches(1024, alloc_position_keys), 
		intensity_fn(intensity_fn), interaction_fn(interaction_fn), 
		intensity_fn_args(intensity_fn_args), interaction_fn_args(interaction_fn_args), 
		n(n), item_type_count(item_type_count), gibbs_iterations(gibbs_iterations) { }

	~map() {
		for (auto entry : patches)
			core::free(entry.value, item_type_count);
	}

	inline float intensity(const position& pos, unsigned int item_type) {
		return intensity_fn(pos, item_type, intensity_fn_args);
	}

	inline float interaction(
			const position& pos1, const position& pos2, unsigned int item_type1, unsigned int item_type2) {
		return interaction_fn(pos1, pos2, item_type1, item_type2, interaction_fn_args);
	}

	inline patch* get_patch_if_exists(const position& patch_position)
	{
		bool contains; unsigned int bucket;
		patches.check_size(alloc_position_keys);
		patch& p = patches.get(patch_position, contains, bucket);
		if (!contains) return NULL;
		else return &p;
	}

	inline patch& get_or_make_patch(const position& patch_position)
	{
		bool contains; unsigned int bucket;
		patches.check_size(alloc_position_keys);
		patch& p = patches.get(patch_position, contains, bucket);
		if (!contains) {
			/* add a new patch */
			init(p, item_type_count);
			patches.table.keys[bucket] = patch_position;
			patches.table.size++;
		}
		return p;
	}

	/**
	 * Returns the patches in the world that intersect with a bounding box of
	 * size n centered at `world_position`. This function will create any
	 * missing patches and ensure that the returned patches are 'fixed': they
	 * cannot be modified in the future by further sampling.
	 */
	void get_fixed_neighborhood(
			position world_position,
			patch* neighborhood[4],
			position patch_positions[4])
	{
		get_neighborhood_positions(world_position, patch_positions);

		for (unsigned int i = 0; i < 4; i++)
			neighborhood[i] = &get_or_make_patch(patch_positions[i]);

		fix_patches(neighborhood, patch_positions, 4);
	}

	/**
	 * Returns the patches in the world that intersect with a bounding box of
	 * size n centered at `world_position`. This function will not create any
	 * missing patches or fix any patches. The number intersecting patches is
	 * returned.
	 */
	unsigned int get_neighborhood(
			position world_position,
			patch* neighborhood[4],
			position patch_positions[4],
			unsigned int& patch_index)
	{
		patch_index = get_neighborhood_positions(world_position, patch_positions);

		unsigned int index = 0;
		for (unsigned int i = 0; i < 4; i++) {
			neighborhood[index] = get_patch_if_exists(patch_positions[i]);
			if (neighborhood[index] != NULL) {
				if (patch_index == i) patch_index = index;
				index++;
			}
		}

		return index;
	}

	void get_items(
			position bottom_left_corner,
			position top_right_corner,
			array<item_position>& items)
	{
		position bottom_left_patch_position, top_right_patch_position;
		world_to_patch_coordinates(bottom_left_corner, bottom_left_patch_position);
		world_to_patch_coordinates(top_right_corner, top_right_patch_position);

		array<patch*> patches(32);
		array<position> patch_positions(32);
		for (int64_t x = bottom_left_patch_position.x; x <= top_right_patch_position.x; x++) {
			for (int64_t y = bottom_left_patch_position.y; y <= top_right_patch_position.y; y++) {
				patch* p = get_patch_if_exists({x, y});
				if (p == NULL) continue;

				for (unsigned int i = 0; i < item_type_count; i++)
					for (position item_position : p->item_positions[i])
						if (item_position.x >= bottom_left_corner.x && item_position.x <= top_right_corner.x
						 && item_position.y >= bottom_left_corner.y && item_position.y <= top_right_corner.y)
							items.add({i, item_position});
			}
		}
	}

	inline void world_to_patch_coordinates(
			position world_position,
			position& patch_position)
	{
		int64_t x_quotient = floored_div(world_position.x, n);
		int64_t y_quotient = floored_div(world_position.y, n);
		patch_position = {x_quotient, y_quotient};
	}

	inline void world_to_patch_coordinates(
			position world_position,
			position& patch_position,
			position& position_within_patch)
	{
		lldiv_t x_quotient = floored_div_with_remainder(world_position.x, n);
		lldiv_t y_quotient = floored_div_with_remainder(world_position.y, n);
		patch_position = {x_quotient.quot, y_quotient.quot};
		position_within_patch = {x_quotient.rem, y_quotient.rem};
	}

private:
	inline int64_t floored_div(int64_t a, unsigned int b) {
		lldiv_t result = lldiv(a, b);
		if (a < 0 && result.rem != 0)
			result.quot--;
		return result.quot;
	}

	inline lldiv_t floored_div_with_remainder(int64_t a, unsigned int b) {
		lldiv_t result = lldiv(a, b);
		if (a < 0 && result.rem != 0) {
			result.quot--;
			result.rem += b;
		}
		return result;
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

	/**
	 * This function ensures that the given patches are fixed: they cannot be
	 * modified in the future by further sampling. New neighboring patches are
	 * created as needed, and sampling is done accordingly.
	 */
	void fix_patches(
			patch** patches,
			const position* patch_positions,
			unsigned int patch_count)
	{
		array<position> positions_to_sample(36);
		for (unsigned int i = 0; i < patch_count; i++) {
			if (patches[i]->fixed) continue;
			positions_to_sample.add(patch_positions[i].up().left());
			positions_to_sample.add(patch_positions[i].up());
			positions_to_sample.add(patch_positions[i].up().right());
			positions_to_sample.add(patch_positions[i].left());
			positions_to_sample.add(patch_positions[i]);
			positions_to_sample.add(patch_positions[i].right());
			positions_to_sample.add(patch_positions[i].down().left());
			positions_to_sample.add(patch_positions[i].down());
			positions_to_sample.add(patch_positions[i].down().right());
			insertion_sort(positions_to_sample);
			unique(positions_to_sample);
		}

		for (unsigned int i = 0; i < positions_to_sample.length; i++) {
			if (get_or_make_patch(positions_to_sample[i]).fixed) {
				positions_to_sample.remove(i);
				i--;
			}
		}

		/* construct the Gibbs field and sample the patches at positions_to_sample */
		gibbs_field<map> field(*this,
				positions_to_sample.data, positions_to_sample.length,
				n, item_type_count);
		for (unsigned int i = 0; i < gibbs_iterations; i++)
			field.sample();

		for (unsigned int i = 0; i < patch_count; i++)
			patches[i]->fixed = true;
	}
};

} /* namespace nel */

#endif /* NEL_MAP_H_ */
