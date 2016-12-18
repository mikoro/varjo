// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <cfloat>
#include <complex>

#include "Common.h"

namespace Varjo
{
	class MathUtils
	{
	public:

		CUDA_CALLABLE static bool almostZero(float value, float threshold = FLT_EPSILON * 4);
		CUDA_CALLABLE static bool almostSame(float first, float second, float threshold = FLT_EPSILON * 4);
		static bool almostSame(const std::complex<float>& first, const std::complex<float>& second, float threshold = FLT_EPSILON * 4);
		CUDA_CALLABLE static float degToRad(float degrees);
		CUDA_CALLABLE static float radToDeg(float radians);
		CUDA_CALLABLE static float smoothstep(float t);
		CUDA_CALLABLE static float smootherstep(float t);
		CUDA_CALLABLE static float fastPow(float a, float b);
	};
}
