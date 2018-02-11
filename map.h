#include "core/map.h"
#include "core/random.h"
#include "math/log.h"
#include "ml/gibbs_field.h"

using namespace core;

namespace map {

struct position {
	int64_t x;
	int64_t y;

	position(int64_t x, int64_t y) : x(x), y(y) { }

	inline position up() const {
		return {x, y + 1};
	}

	inline position down() const {
		return {x, y - 1};
	}

	inline position left() const {
		return {x - 1, y};
	}

	inline position right() const {
		return {x + 1, y};
	}

	inline position operator * (unsigned int k) const {
		return {x * k, y * k};
	}

	inline position operator + (const position& p) const {
		return {x + p.x, y + p.y};
	}

	inline position operator - (const position& p) const {
		return {x - p.x, y - p.y};
	}

	inline bool operator < (const position& p) const {
		if (x < p.x) return true;
		else if (x > p.x) return false;
		else if (y < p.y) return true;
		else return false;
	}
};

template<unsigned int n, unsigned int ItemTypeCount>
class patch {
	array<position> item_positions[ItemTypeCount];

	/* indicates if this patch is fixed, or if it can
	   be resampled (for example, if it's on the edge) */
	bool fixed;

public:
};

template<unsigned int n, unsigned int ItemTypeCount>
inline bool init(patch<n, ItemTypeCount>& new_patch) {
	new_patch.fixed = false;
	for (unsigned int i = 0; i < ItemTypeCount; i++) {
		if (!array_init(new_patch.item_positions[i], 8)) {
			for (unsigned int j = 0; j < i; j++) free(new_patch.item_positions[j]);
			return false;
		}
	}
	return true;
}

template<unsigned int n, unsigned int ItemTypeCount>
inline void free(patch<n, ItemTypeCount>& p) {
	for (unsigned int i = 0; i < ItemTypeCount; i++)
		free(p.item_positions[i]);
}

template<unsigned int n, unsigned int ItemTypeCount,
	typename IntensityFunction, typename InteractionFunction>
struct map {
	hash_map<unsigned int, hash_map<unsigned int, patch<n, ItemTypeCount>>> patches;

	static constexpr unsigned int patch_size = n;

	typedef patch<n, ItemTypeCount> patch_type;
	typedef map<n, ItemTypeCount, IntensityFunction, InteractionFunction> map_type;

public:
	map() : patches(1024) { }

	inline position world_to_patch_position(const position& world_position) const {
		return {world_position.x / n, world_position.y / n};
	}

	inline patch<n, ItemTypeCount>* get_patch_if_exists(const position& patch_position)
	{
		bool contains; unsigned int bucket;
		patches.check_size();
		const auto& column = patches.get(patch_position.x, contains, bucket);
		if (!exists) return false;

		column.check_size();
		patch<n, ItemTypeCount>& p = column.get(patch_position.y, contains, bucket);
		if (!contains) return false;
		return &p;
	}

	inline patch<n, ItemTypeCount>& get_or_make_patch(const position& patch_position)
	{
		bool contains, unsigned int bucket;
		patches.check_size();
		const auto& column = patches.get(patch_position.x, contains, bucket);
		if (!contains) {
			/* add a new column */
			hash_map_init(column, 16);
			patches.table.keys[bucket] = patch_position.x;
			patches.table.size++;
		}

		column.check_size();
		patch<n, ItemTypeCount>& p = column.get(patch_position.y, contains, bucket);
		if (!contains) {
			/* add a new patch */
			init(p);
			column.table.keys[bucket] = patch_position.y;
			column.table.size++;
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
			patch<n, ItemTypeCount>* neighborhood[4],
			position patch_positions[4])
	{
		get_neighborhood_positions(world_position, neighborhood, patch_positions);

		for (unsigned int i = 0; i < 4; i++)
			patches[i] = get_or_make_patch(patch_positions[i]);

		fix_patches(patch_positions);
	}

	/**
	 * Returns the patches in the world that intersect with a bounding box of
	 * size n centered at `world_position`. This function will not create any
	 * missing patches or fix any patches. The number intersecting patches is
	 * returned.
	 */
	unsigned int get_neighborhood(
			position world_position,
			patch<n, ItemTypeCount>* neighborhood[4],
			position patch_positions[4],
			unsigned int& patch_index)
	{
		patch_index = get_neighborhood_positions(world_position, neighborhood, patch_positions);

		unsigned int index = 0;
		for (unsigned int i = 0; i < 4; i++) {
			patches[index] = get_patch_if_exists(patch_positions[i]);
			if (patches[index] != NULL) index++;
		}

		return index;
	}

private:
	unsigned int get_neighborhood_positions(
			position world_position,
			patch<n, ItemTypeCount>* neighborhood[4],
			position patch_positions[4])
	{
		position patch_position = {world_position.x / n, world_position.y / n};
		position relative_position = {world_position.x % n, world_position.y % n};

		/* determine the quadrant of our current location in current_patch */
		unsigned int patch_index;
		if (relative_position.x < (n / 2)) {
			/* we are in the left half of this patch */
			if (relative_position.y < (n / 2)) {
				/* we are in the bottom-left quadrant */
				patch_positions[0] = patch_position.left();
				patch_index = 1;
			} else {
				/* we are in the top-left quadrant */
				patch_positions[0] = patch_position.left().up();
				patch_index = 2;
			}
		} else {
			/* we are in the right half of this patch */
			if (relative_position.y < (n / 2)) {
				/* we are in the bottom-right quadrant */
				patch_positions[0] = patch_position;
				patch_index = 0;
			} else {
				/* we are in the top-right quadrant */
				patch_positions[0] = patch_position.up();
				patch_index = 3;
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
	template<size_t N>
	void fix_patches(
			patch<n, ItemTypeCount> (&patches)[N],
			const position (&patch_positions)[N])
	{
		array<position> positions_to_sample(N * 9);
		for (unsigned int i = 0; i < N; i++) {
			if (patches[i].fixed) continue;
			positions_to_sample.add(patch_positions[i].up().left());
			positions_to_sample.add(patch_positions[i].up());
			positions_to_sample.add(patch_positions[i].up().right());
			positions_to_sample.add(patch_positions[i].left());
			positions_to_sample.add(patch_positions[i]);
			positions_to_sample.add(patch_positions[i].right());
			positions_to_sample.add(patch_positions[i].down().left());
			positions_to_sample.add(patch_positions[i].down());
			positions_to_sample.add(patch_positions[i].down().right());
			insertion_sort(to_sample);
			unique(to_sample);
		}

		for (unsigned int i = 0; i < positions_to_sample.length; i++) {
			if (get_or_make_patch(positions_to_sample[i]).fixed) {
				positions_to_sample.remove(i);
				i--;
			}
		}

		/* construct the Gibbs field and sample the patches at positions_to_sample */
		gibbs_field<map_type, IntensityFunction, InteractionFunction> field(
			*this, positions_to_sample.data, positions_to_sample.length);
		/* TODO: do the actual sampling */

		for (unsigned int i = 0; i < N; i++)
			patches[i].fixed = true;
	}
};

template<unsigned int n, unsigned int ItemTypeCount,
	typename IntensityFunction, typename InteractionFunction>
inline void sample_cell(
		map<n, ItemTypeCount>& m,
		const position& patch_position,
		const position& position_within_patch)
{
	patch<n, ItemTypeCount>* neighborhood[4];
	position neighbor_positions[4];
	position world_position = patch_position * n + position_within_patch;

	unsigned int patch_index;
	unsigned int neighbor_count = m.get_neighborhood(world_position, neigborhood, neighbor_positions, patch_index);

	double log_probabilities[ItemTypeCount + 1];
	unsigned int old_item_type = ItemTypeCount;
	for (unsigned int i = 0; i < ItemTypeCount; i++) {
		/* compute the energy contribution of this cell when the item type is `i` */
		log_probabilities[i] = IntensityFunction(world_position, i);
		for (unsigned int j = 0; j < neighbor_count; j++) {
			for (unsigned int k = 0; k < ItemTypeCount; k++) {
				for (position item_position : neighborhood[j]->item_positions[k]) {
					if (item_position == world_position) {
						old_item_type = k; continue; /* ignore the current position */
					}

					log_probabilities[i] += InteractionFunction(world_position, item_position, i, k);
				}
			}
		}
	}

	log_probabilities[ItemTypeCount] = 0.0;
	normalize_exp(log_probabilities, ItemTypeCount);
	unsigned int sampled_item_type = sample_categorical(log_probabilities, ItemTypeCount);

	patch<n, ItemTypeCount>& current_patch = *neighborhood[patch_index];
	if (old_item_type == sampled_item_type) {
		/* the Gibbs step didn't change anything */
		return;
	} if (old_item_type < ItemTypeCount) {
		/* find and remove the old item position */
		for (unsigned int i = 0; i < current_patch.item_positions[old_item_type].length; i++) {
			if (current_patch.item_positions[old_item_type][i] == world_position) {
				current_patch.item_positions[old_item_type].remove(i);
				break;
			}
		}
	} if (sampled_item_type < ItemTypeCount) {
		current_patch.item_positions[sampled_item_type].add(world_position);
	}
}

}; /* namespace map */
