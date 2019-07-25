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

#ifndef DIFFUSION_H_
#define DIFFUSION_H_

#include <core/core.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

namespace jbw {

using namespace core;

template<typename T>
struct diffusion
{
	unsigned int radius;
	unsigned int max_time;
	T alpha; /* diffusion constant */
	T lambda; /* decay constant */

	/* the cache of pre-computed states */
	T** cache;

	~diffusion() { free_helper(); }

	inline T get_value(unsigned int t, int x, int y) const {
		if (x < 0) return get_value(t, -x, y);
		if (y < 0) return get_value(t, x, -y);
		if (y > x) return get_value(t, y, x);
#if !defined(NDEBUG)
		if ((unsigned int) x >= radius)
			fprintf(stderr, "diffusion.get_value WARNING: Requested position "
					"is beyond the radius of this diffusion simulation.\n");
		if (t >= max_time)
			fprintf(stderr, "diffusion.get_value WARNING: Requested time (%u) "
					"is beyond the bounds of this diffusion simulation.\n", t);
#endif
		return cache[t][(x * (x + 1)) / 2 + y];
	}

	static inline void free(diffusion<T>& model) {
		model.free_helper();
	}

private:
	inline void free_helper() {
		for (unsigned int t = 0; t < max_time; t++)
			core::free(cache[t]);
		core::free(cache);
	}
};

template<typename T>
bool init(diffusion<T>& model, T alpha, T lambda,
		unsigned int patch_size, unsigned int max_time)
{
	if (fabs(lambda) + 4 * fabs(alpha) >= 1.0) {
		fprintf(stderr, "init ERROR: The diffusion model is divergent"
				" for the given alpha and lambda parameters.\n");
		return false;
	}

	unsigned int radius = max(patch_size / 2 + 1, 1u);
	model.max_time = max_time;
	model.alpha = alpha;
	model.lambda = lambda;
	model.radius = radius;

	model.cache = (T**) malloc(sizeof(T*) * max_time);
	if (model.cache == NULL) {
		fprintf(stderr, "init ERROR: Insufficient memory for diffusion.cache.\n");
		return false;
	}
	unsigned int cache_entry_size = ((radius * (radius + 1)) / 2);
	for (unsigned int t = 0; t < max_time; t++) {
		model.cache[t] = (T*) malloc(sizeof(T) * cache_entry_size);
		if (model.cache[t] == NULL) {
			fprintf(stderr, "init ERROR: Insufficient memory for diffusion.cache[%u].\n", t);
			for (unsigned int j = 0; j < t; j++) free(model.cache[j]);
			free(model.cache); return false;
		}
	}

	/* run the simulation in the cache */
	memset(model.cache[0], 0, cache_entry_size * sizeof(T));
	model.cache[0][0] = (T) 1.0;
	for (unsigned int t = 1; t < max_time; t++) {
		/* decay the value from the previous time step */
		for (unsigned int i = 0; i < cache_entry_size; i++)
			model.cache[t][i] = lambda * model.cache[t - 1][i];

		/* add new value at origin */
		model.cache[t][0] += (T) 1.0;

		/* first diffuse the corner and edge */
		model.cache[t][cache_entry_size - 1] += 2 * alpha * model.cache[t - 1][cache_entry_size - 2];
		for (unsigned int y = 0; y + 1 < radius; y++)
			model.cache[t][cache_entry_size - radius + y] +=
					alpha * (model.get_value(t - 1, radius - 2, y)
						+ model.get_value(t - 1, radius - 1, y + 1)
						+ model.get_value(t - 1, radius - 1, y - 1));

		/* diffuse the interior */
		for (unsigned int x = 0; x + 1 < radius; x++)
			for (unsigned int y = 0; y <= x; y++)
				model.cache[t][((x + 1) * x) / 2 + y] +=
						alpha * (model.get_value(t - 1, x + 1, y)
							+ model.get_value(t - 1, x - 1, y)
							+ model.get_value(t - 1, x, y + 1)
							+ model.get_value(t - 1, x, y - 1));
	}
	return true;
}

} /* namespace jbw */

#endif /* DIFFUSION_H_ */
