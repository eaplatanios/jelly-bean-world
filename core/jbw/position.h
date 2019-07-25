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

#ifndef JBW_POSITION_H_
#define JBW_POSITION_H_

#include <core/map.h>
#include <inttypes.h>

namespace jbw {

using namespace core;

struct position {
	int64_t x;
	int64_t y;

	position() { }

	explicit constexpr position(int64_t v) : x(v), y(v) { }

	constexpr position(int64_t x, int64_t y) : x(x), y(y) { }

	position(const position& src) : x(src.x), y(src.y) { }

	inline uint64_t squared_length() const {
		return (uint64_t) x*x + (uint64_t) y*y;
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

	inline position& operator += (const position& p) {
		x += p.x; y += p.y; return *this;
	}

	inline position& operator -= (const position& p) {
		x -= p.x; y -= p.y; return *this;
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

	static inline void swap(position& first, position& second) {
		core::swap(first.x, second.x);
		core::swap(first.y, second.y);
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

	static inline void set_empty(position* p, unsigned int length) {
		for (unsigned int i = 0; i < length; i++)
			set_empty(p[i]);
	}

	static constexpr int64_t MAX_INT64 = std::numeric_limits<int64_t>::max();
};

template<typename Stream>
inline bool read(position& p, Stream& in) {
	return read(p.x, in) && read(p.y, in);
}

template<typename Stream>
inline bool write(const position& p, Stream& out) {
	return write(p.x, out) && write(p.y, out);
}

template<typename Stream>
inline bool print(const position& p, Stream& out) {
	return fprintf(out, "(%" PRId64 ", %" PRId64 ")", p.x, p.y) > 0;
}

} /* namespace jbw */

#endif /* JBW_POSITION_H_ */
