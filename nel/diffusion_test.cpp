#include "diffusion.h"

#include <core/io.h>

using namespace core;
using namespace nel;

template<typename T>
void test_diffusion(T alpha, T lambda,
		unsigned int patch_size, unsigned int max_time)
{
	diffusion<T> model;
	bool result = init(model, alpha, lambda, patch_size, max_time);
	if (!result) return;

	int radius = max((int) (patch_size / 2) + 1, 1);
	for (unsigned int t = 0; t < max_time; t++) {
		fprintf(stderr, "t = %u: %.20lf at (0,0), %.20lf at (%d,0), %.20lf at (%d,%d)\n", t,
				model.get_value(t, 0, 0), model.get_value(t, radius - 1, 0), radius - 1,
				model.get_value(t, radius - 1, radius - 1), radius - 1, radius - 1);
	}
}

int main(int argc, const char** argv) {
	test_diffusion<double>(0.12, 0.5, 64, 2000 + 1);
	return EXIT_SUCCESS;
}
