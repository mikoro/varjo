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

__device__ inline void writeToPixels(const Paths* __restrict paths, Pixel* __restrict pixels, uint32_t pathId, uint32_t filmWidth, uint32_t filmHeight)
{
	int ox = int(paths->filmSamplePosition[pathId].x);
	int oy = int(paths->filmSamplePosition[pathId].y);

	for (int tx = -1; tx <= 2; ++tx)
	{
		for (int ty = -1; ty <= 2; ++ty)
		{
			int px = ox + tx;
			int py = oy + ty;
			px = clamp(px, 0, int(filmWidth) - 1);
			py = clamp(py, 0, int(filmHeight) - 1);
			float2 pixelPosition = make_float2(float(px), float(py));
			float2 distance = pixelPosition - paths->filmSamplePosition[pathId];
			float weight = mitchellFilter(distance.x) * mitchellFilter(distance.y);
			float3 color = weight * paths->result[pathId];
			int pixelIndex = py * int(filmWidth) + px;

			atomicAdd(&(pixels[pixelIndex].color.x), color.x);
			atomicAdd(&(pixels[pixelIndex].color.y), color.y);
			atomicAdd(&(pixels[pixelIndex].color.z), color.z);
			atomicAdd(&(pixels[pixelIndex].weight), weight);
		}
	}
}
