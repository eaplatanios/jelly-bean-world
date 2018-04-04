#ifndef NEL_MAP_H_
#define NEL_MAP_H_

#include <core/map.h>
#include "gibbs_field.h"

namespace nel {

using namespace core;

struct item {
	unsigned int item_type;

	/* the position of the item, in world coordinates */
	position location;

	/* a time of 0 indicates the item always existed */
	uint64_t creation_time;
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

void* alloc_position_keys(size_t n, size_t element_size) {
	position* keys = (position*) malloc(sizeof(position) * n);
	if (keys == NULL) return NULL;
	for (unsigned int i = 0; i < n; i++)
		position::set_empty(keys[i]);
	return (void*) keys;
}

template<typename PerPatchData>
struct map {
	hash_map<position, patch<PerPatchData>> patches;

	intensity_function intensity_fn;
	interaction_function interaction_fn;

	// We assume that the length of the args arrays is known at this point and has been checked.
	float* intensity_fn_args;
	float* interaction_fn_args;

	unsigned int n;
	unsigned int item_type_count;
	unsigned int gibbs_iterations;

	std::minstd_rand rng;
	gibbs_field_cache cache;

	typedef patch<PerPatchData> patch_type;
	typedef item item_type;

public:
	map(unsigned int n, unsigned int item_type_count, unsigned int gibbs_iterations,
			intensity_function intensity_fn, float* intensity_fn_args, 
			interaction_function interaction_fn, float* interaction_fn_args, uint_fast32_t seed) :
		patches(1024, alloc_position_keys), 
		intensity_fn(intensity_fn), interaction_fn(interaction_fn), 
		intensity_fn_args(intensity_fn_args), interaction_fn_args(interaction_fn_args), 
		n(n), item_type_count(item_type_count), gibbs_iterations(gibbs_iterations),
		cache(*this, n, item_type_count, is_stationary(intensity_fn), is_stationary(interaction_fn))
	{
		rng.seed(seed);
#if !defined(NDEBUG)
		rng.seed(0);
#else
		rng.seed(milliseconds());
#endif
	}

	map(unsigned int n, unsigned int item_type_count, unsigned int gibbs_iterations,
			intensity_function intensity_fn, float* intensity_fn_args, 
			interaction_function interaction_fn, float* interaction_fn_args) :
		map(n, item_type_count, gibbs_iterations, intensity_fn, intensity_fn_args, interaction_fn, interaction_fn_args,
#if !defined(NDEBUG)
			0) { }
#else
			milliseconds()) { }
#endif

	~map() { free_helper(); }

	inline void set_seed(uint_fast32_t new_seed) {
		rng.seed(new_seed);
	}

	inline float intensity(const position& pos, unsigned int item_type) const {
		return intensity_fn(pos, item_type, intensity_fn_args);
	}

	inline float interaction(
			const position& pos1, const position& pos2,
			unsigned int item_type1, unsigned int item_type2) const
	{
		return interaction_fn(pos1, pos2, item_type1, item_type2, interaction_fn_args);
	}

	inline patch_type& get_existing_patch(const position& patch_position) {
#if !defined(NDEBUG)
		bool contains;
		patch_type& patch = patches.get(patch_position, contains);
		if (!contains) fprintf(stderr, "map.get_existing_patch WARNING: The requested patch does not exist.\n");
		return patch;
#else
		return patches.get(patch_position);
#endif
	}

	inline patch_type* get_patch_if_exists(const position& patch_position) const
	{
		bool contains;
		patch_type& p = patches.get(patch_position, contains);
		if (!contains) return NULL;
		else return &p;
	}

	template<bool ResizeMap = true>
	inline patch_type& get_or_make_patch(const position& patch_position)
	{
		bool contains; unsigned int bucket;
		if (ResizeMap) patches.check_size(alloc_position_keys);
		patch_type& p = patches.get(patch_position, contains, bucket);
		if (!contains) {
			/* add a new patch */
			init(p);
			patches.table.keys[bucket] = patch_position;
			patches.table.size++;
		}
		return p;
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
			position patch_positions[4])
	{
		unsigned int index = get_neighborhood_positions(world_position, patch_positions);

		patches.check_size(patches.table.size + 16, alloc_position_keys);
		for (unsigned int i = 0; i < 4; i++)
			neighborhood[i] = &get_or_make_patch<false>(patch_positions[i]);

		fix_patches(neighborhood, patch_positions, 4);
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

	template<typename ProcessNeighborhood>
	void iterate_neighborhoods(const position& patch_position, ProcessNeighborhood process_neighborhood_function)
	{
		patch_type* current = get_patch_if_exists(patch_position);
		patch_type* top = get_patch_if_exists(patch_position.up());
		patch_type* bottom = get_patch_if_exists(patch_position.down());
		patch_type* left = get_patch_if_exists(patch_position.left());
		patch_type* right = get_patch_if_exists(patch_position.right());
		patch_type* top_left = get_patch_if_exists(patch_position.up().left());
		patch_type* top_right = get_patch_if_exists(patch_position.up().right());
		patch_type* bottom_left = get_patch_if_exists(patch_position.down().left());
		patch_type* bottom_right = get_patch_if_exists(patch_position.down().right());

		patch_type* bottom_left_neighborhood[4];
		patch_type* top_left_neighborhood[4];
		patch_type* bottom_right_neighborhood[4];
		patch_type* top_right_neighborhood[4];
		unsigned int bottom_left_neighbor_count = 1;
		unsigned int top_left_neighbor_count = 1;
		unsigned int bottom_right_neighbor_count = 1;
		unsigned int top_right_neighbor_count = 1;
		bottom_left_neighborhood[0] = current;
		top_left_neighborhood[0] = current;
		bottom_right_neighborhood[0] = current;
		top_right_neighborhood[0] = current;
		if (left != NULL) {
			bottom_left_neighborhood[bottom_left_neighbor_count++] = left;
			top_left_neighborhood[top_left_neighbor_count++] = left;
		} if (right != NULL) {
			bottom_right_neighborhood[bottom_right_neighbor_count++] = right;
			top_right_neighborhood[top_right_neighbor_count++] = right;
		} if (top != NULL) {
			top_left_neighborhood[top_left_neighbor_count++] = top;
			top_right_neighborhood[top_right_neighbor_count++] = top;
		} if (bottom != NULL) {
			bottom_left_neighborhood[bottom_left_neighbor_count++] = bottom;
			bottom_right_neighborhood[bottom_right_neighbor_count++] = bottom;
		} if (bottom_left != NULL) {
			bottom_left_neighborhood[bottom_left_neighbor_count++] = bottom_left;
		} if (top_left != NULL) {
			top_left_neighborhood[top_left_neighbor_count++] = top_left;
		} if (bottom_right != NULL) {
			bottom_right_neighborhood[bottom_right_neighbor_count++] = bottom_right;
		} if (top_right != NULL) {
			top_right_neighborhood[top_right_neighbor_count++] = top_right;
		}

		unsigned int half_n = n / 2;
		unsigned int x, y;
		for (x = 0; x < half_n; x++) {
			for (y = 0; y < half_n; y++) {
				/* bottom-left quadrant */
				process_neighborhood_function(x, y, bottom_left_neighborhood, bottom_left_neighbor_count);
			} for (; y < n; y++) {
				/* top-left quadrant */
				process_neighborhood_function(x, y, top_left_neighborhood, top_left_neighbor_count);
			}
		} for (; x < n; x++) {
			for (y = 0; y < half_n; y++) {
				/* bottom-right quadrant */
				process_neighborhood_function(x, y, bottom_right_neighborhood, bottom_right_neighbor_count);
			} for (; y < n; y++) {
				/* top-right quadrant */
				process_neighborhood_function(x, y, top_right_neighborhood, top_right_neighbor_count);
			}
		}
	}

	template<typename ProcessPatchFunction>
	bool get_state(
			position bottom_left_corner,
			position top_right_corner,
			ProcessPatchFunction process_patch,
			position& bottom_left_patch_position,
			position& top_right_patch_position) const
	{
		world_to_patch_coordinates(bottom_left_corner, bottom_left_patch_position);
		world_to_patch_coordinates(top_right_corner, top_right_patch_position);

		array<patch_type*> patches(32);
		array<position> patch_positions(32);
		for (int64_t x = bottom_left_patch_position.x; x <= top_right_patch_position.x; x++) {
			for (int64_t y = bottom_left_patch_position.y; y <= top_right_patch_position.y; y++) {
				patch_type* p = get_patch_if_exists({x, y});
				if (p != NULL && !process_patch(*p, position(x, y)))
					return false;
			}
		}
		return true;
	}

	template<typename ProcessPatchFunction>
	inline bool get_state(
			position bottom_left_corner,
			position top_right_corner,
			ProcessPatchFunction process_patch) const
	{
		position bottom_left_patch_position, top_right_patch_position;
		return get_state(bottom_left_corner, top_right_corner, process_patch,
				bottom_left_patch_position, top_right_patch_position);
	}

	bool get_items(
			position bottom_left_corner,
			position top_right_corner,
			array<item>& items) const
	{
		auto process_patch = [&](const patch_type& p, position patch_position) {
			for (const item& i : p.items)
				if (i.location.x >= bottom_left_corner.x && i.location.x <= top_right_corner.x
				 && i.location.y >= bottom_left_corner.y && i.location.y <= top_right_corner.y
				 && !items.add(i))
					return false;
			return true;
		};
		return get_state(bottom_left_corner, top_right_corner, process_patch);
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
	 *
	 * NOTE: This function assumes `patches` has sufficient capacity to store
	 * any new patches that may be initialized.
	 */
	void fix_patches(
			patch_type** patches,
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
			if (get_or_make_patch<false>(positions_to_sample[i]).fixed) {
				positions_to_sample.remove(i);
				i--;
			}
		}

		/* construct the Gibbs field and sample the patches at positions_to_sample */
		if (is_stationary(intensity_fn)) {
			if (is_stationary(interaction_fn)) {
				sample_patches<true, true>(positions_to_sample);
			} else {
				sample_patches<true, false>(positions_to_sample);
			}
		} else {
			if (is_stationary(interaction_fn)) {
				sample_patches<false, true>(positions_to_sample);
			} else {
				sample_patches<false, false>(positions_to_sample);
			}
		}

		for (unsigned int i = 0; i < patch_count; i++)
			patches[i]->fixed = true;
	}

	template<bool StationaryIntensity, bool StationaryInteraction>
	inline void sample_patches(const array<position>& positions_to_sample) {
		gibbs_field<map<PerPatchData>, StationaryIntensity, StationaryInteraction> field(
				*this, cache, positions_to_sample.data, (unsigned int) positions_to_sample.length, n, item_type_count);
		for (unsigned int i = 0; i < gibbs_iterations; i++)
			field.sample(rng);
	}

	inline void free_helper() {
		for (auto entry : patches)
			core::free(entry.value);
	}
};

template<typename PerPatchData>
inline bool init(map<PerPatchData>& world, unsigned int n,
		unsigned int item_type_count, unsigned int gibbs_iterations,
		intensity_function intensity_fn, float* intensity_fn_args, 
		interaction_function interaction_fn, float* interaction_fn_args,
		uint_fast32_t seed)
{
	if (!hash_map_init(world.patches, 1024, alloc_position_keys))
		return false;
	world.intensity_fn = intensity_fn;
	world.interaction_fn = interaction_fn;
	world.intensity_fn_args = intensity_fn_args;
	world.interaction_fn_args = interaction_fn_args;
	world.n = n;
	world.item_type_count = item_type_count;
	world.gibbs_iterations = gibbs_iterations;
	if (!init(world.cache, world, n, item_type_count,
			is_stationary(intensity_fn), is_stationary(interaction_fn))) {
		free(world.patches);
		return false;
	}

	new (&world.rng) std::minstd_rand(seed);
	return true;
}

template<typename PerPatchData>
inline bool init(map<PerPatchData>& world, unsigned int n,
		unsigned int item_type_count, unsigned int gibbs_iterations,
		intensity_function intensity_fn, float* intensity_fn_args, 
		interaction_function interaction_fn, float* interaction_fn_args)
{
#if !defined(NDEBUG)
	uint_fast32_t seed = 0;
#else
	uint_fast32_t seed = (uint_fast32_t) milliseconds();
#endif
	return init(world, n, item_type_count, gibbs_iterations, intensity_fn,
			intensity_fn_args, interaction_fn, interaction_fn_args, seed);
}

template<typename PerPatchData, typename Stream, typename PatchReader>
bool read(map<PerPatchData>& world, Stream& in,
		intensity_function intensity_fn,
		interaction_function interaction_fn,
		float* intensity_fn_args,
		float* interaction_fn_args,
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

	default_scribe scribe;
	world.intensity_fn = intensity_fn;
	world.interaction_fn = interaction_fn;
	world.intensity_fn_args = intensity_fn_args;
	world.interaction_fn_args = interaction_fn_args;
	if (!read(world.n, in)
	 || !read(world.item_type_count, in)
	 || !read(world.gibbs_iterations, in)
	 || !read(world.patches, in, alloc_position_keys, scribe, patch_reader))
		return false;
	if (!init(world.cache, world, world.n, world.item_type_count,
			is_stationary(intensity_fn), is_stationary(interaction_fn)))
	{
		free(world.patches);
		return false;
	}
	return true;
}

/* NOTE: this function assumes the variables in the map are not modified during writing */
template<typename PerPatchData, typename Stream, typename PatchWriter>
bool write(const map<PerPatchData>& world, Stream& out,
		PatchWriter& patch_writer = default_scribe())
{
	/* write the PRNG state into a stringstream buffer */
	std::stringstream buffer;
	buffer << world.rng;
	std::string data = buffer.str();
	if (!write(data.length(), out)
	 || !write(data.c_str(), out, (unsigned int) data.length()))
		return false;

	default_scribe scribe;
	return write(world.n, out)
		&& write(world.item_type_count, out)
		&& write(world.gibbs_iterations, out)
		&& write(world.patches, out, scribe, patch_writer);
}

} /* namespace nel */

#endif /* NEL_MAP_H_ */
