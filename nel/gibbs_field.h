#ifndef NEL_GIBBS_FIELD_H_
#define NEL_GIBBS_FIELD_H_

#include "position.h"

namespace nel {

template<typename Map, typename IntensityFunction, typename InteractionFunction>
class gibbs_field
{
	Map& map;
	position* patch_positions;
	unsigned int patch_count;

	static constexpr unsigned int n = Map::patch_size;
	typedef Map::patch_type patch_type;

public:
	/* NOTE: `patches` and `patch_positions` are used directly, and not copied */
	gibbs_field(Map& map, position* patch_positions, unsigned int patch_count) :
			map(map), patch_positions(patch_positions), patch_count(patch_count) { }

	~gibbs_field() { }

	void sample() {
		for (unsigned int i = 0; i < patch_count; i++)
			sample_patch<IntensityFunction, InteractionFunction>(patch_positions[i]);
	}

private:
	inline void sample_patch(const position& patch_position) {
		for (unsigned int x = 0; x < n; x++)
			for (unsigned int y = 0; y < n; y++)
				sample_cell<IntensityFunction, InteractionFunction>(map, patch_position, {x, y});
	}
};

} /* namespace nel */

#endif /* NEL_GIBBS_FIELD_H_ */
