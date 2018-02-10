#include <stdlib.h>
#include <core/map.h>

using namespace core;

template<typename T, unsigned int n>
class patch {
	T* value;

public:
};

struct position {
	unsigned int x;
	unsigned int y;
};

template<typename T, unsigned int n>
struct map {
	hash_map<unsigned int, hash_map<unsigned int, patch<T, n>>> patches;

	static constexpr unsigned int patch_size = n;

	typedef T cell_type;

public:
	map() : patches(1024) {
		
	}

	inline position world_to_patch_position(const position& world_position) const {
		return {world_position.x / n, world_position.y / n};
	}

	inline patch<T, n>& get_patch(const position& patch_position) {
#if !defined(NDEBUG)
		bool contains;
		const auto& column = patches.get(patch_position.x, contains);
		if (!contains) {
			fprintf(stderr, "map.get_patch ERROR: Patch is not in the map.\n");
			exit(EXIT_FAILURE);
		}

		patch<T, n>& p = column.get(patch_position.y, contains);
		if (!contains) {
			fprintf(stderr, "map.get_patch ERROR: Patch is not in the map.\n");
			exit(EXIT_FAILURE);
		}
		return p;
#else
		return patches.get(patch_position.x).get(patch_position.y);
#endif
	}
};
