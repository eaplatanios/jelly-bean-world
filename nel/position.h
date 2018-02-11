#ifndef NEL_POSITION_H_
#define NEL_POSITION_H_

#include <core/map.h>

using namespace core;

namespace nel {

struct position {
	int64_t x;
	int64_t y;

	position() { }

	explicit position(int64_t v) : x(v), y(v) { }

	position(int64_t x, int64_t y) : x(x), y(y) { }

	position(const position& src) : x(src.x), y(src.y) { }

	inline uint64_t squared_length() const {
		return x*x + y*y;
	}

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

	template<typename V, typename std::enable_if<std::is_arithmetic<V>::value>::type* = nullptr>
	inline position operator * (V k) const {
		return {x * k, y * k};
	}

	inline position operator + (const position& p) const {
		return {x + p.x, y + p.y};
	}

	inline position operator - (const position& p) const {
		return {x - p.x, y - p.y};
	}

	inline bool operator == (const position& p) const {
		return x == p.x && y == p.y;
	}

	inline bool operator != (const position& p) const {
		return x != p.x || y != p.y;
	}

	inline bool operator < (const position& p) const {
		if (x < p.x) return true;
		else if (x > p.x) return false;
		else if (y < p.y) return true;
		else return false;
	}

	static inline void move(const position& src, position& dst) {
		dst.x = src.x; dst.y = src.y;
	}

	static inline unsigned int hash(const position& key) {
		return default_hash(key.x) ^ default_hash(key.y);
	}

	static inline bool is_empty(const position& p) {
		return p.x == MAX_INT64 && p.y == MAX_INT64;
	}

	static inline void set_empty(position& p) {
		p.x = MAX_INT64;
		p.y = MAX_INT64;
	}

	static constexpr int64_t MAX_INT64 = std::numeric_limits<int64_t>::max();
};

} /* namespace nel */

template<typename Stream>
inline bool print(const nel::position& p, Stream& out) {
	return fprintf(out, "(%ld, %ld)", p.x, p.y) > 0;
}

#endif /* NEL_POSITION_H_ */
