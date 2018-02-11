#ifndef NEL_POSITION_H_
#define NEL_POSITION_H_

#include <cstdint>

namespace nel {

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

} /* namespace nel */

#endif /* NEL_POSITION_H_ */
