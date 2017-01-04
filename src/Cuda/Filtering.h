// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <cmath>
#include <cuda_runtime.h>

__device__ inline float mitchellFilter(float s)
{
	const float B = 1.0f / 3.0f;
	const float C = 1.0f / 3.0f;

	s = fabsf(s);

	if (s < 1.0f)
		return ((12.0f - 9.0f * B - 6.0f * C) * (s * s * s) + (-18.0f + 12.0f * B + 6.0f * C) * (s * s) + (6.0f - 2.0f * B)) * (1.0f / 6.0f);

	if (s < 2.0f)
		return ((-B - 6.0f * C) * (s * s * s) + (6.0f * B + 30.0f * C) * (s * s) + (-12.0f * B - 48.0f * C) * s + (8.0f * B + 24.0f * C)) * (1.0f / 6.0f);

	return 0.0f;
}
