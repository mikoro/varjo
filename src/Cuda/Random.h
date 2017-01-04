// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#include "Cuda/Renderer.h"

using namespace Varjo;

// http://www.pcg-random.org/
__device__ inline uint32_t randomInt(Random& random)
{
	uint64_t oldstate = random.state;
	random.state = oldstate * 6364136223846793005ULL + random.inc;
	uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
	uint32_t rot = oldstate >> 59u;
	return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

__device__ inline uint32_t randomInt(Random& random, uint32_t max)
{
	uint32_t threshold = -max % max;

	for (;;)
	{
		uint32_t r = randomInt(random);

		if (r >= threshold)
			return r % max; // 0 <= r < max
	}
}

__device__ inline float randomFloat(Random& random)
{
	//return float(ldexp(randomInt(random), -32)); // http://mumble.net/~campbell/tmp/random_real.c
	return float(randomInt(random)) / float(0xFFFFFFFF);
}

__device__ inline void initRandom(Random& random, uint64_t initstate, uint64_t initseq)
{
	random.state = 0U;
	random.inc = (initseq << 1u) | 1u;
	randomInt(random);
	random.state += initstate;
	randomInt(random);
}
