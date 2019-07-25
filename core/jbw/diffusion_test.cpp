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

#include "diffusion.h"

#include <core/io.h>

using namespace core;
using namespace jbw;

template<typename T>
void test_diffusion(T alpha, T lambda,
		unsigned int patch_size, unsigned int max_time)
{
	diffusion<T>& model = *((diffusion<T>*) alloca(sizeof(diffusion<T>)));
	bool result = init(model, alpha, lambda, patch_size, max_time);
	if (!result) return;

	int radius = max((int) (patch_size / 2) + 1, 1);
	for (unsigned int t = 0; t < max_time; t++) {
		fprintf(stderr, "t = %u: %.20lf at (0,0), %.20lf at (%d,0), %.20lf at (%d,%d)\n", t,
				model.get_value(t, 0, 0), model.get_value(t, radius - 1, 0), radius - 1,
				model.get_value(t, radius - 1, radius - 1), radius - 1, radius - 1);
	}
	free(model);
}

int main(int argc, const char** argv) {
	test_diffusion<double>(0.14, 0.4, 32, 2000 + 1);
	return EXIT_SUCCESS;
}
