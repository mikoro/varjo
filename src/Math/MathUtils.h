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

		static bool almostZero(float value, float threshold = FLT_EPSILON * 4);
		static bool almostSame(float first, float second, float threshold = FLT_EPSILON * 4);
		static bool almostSame(const std::complex<float>& first, const std::complex<float>& second, float threshold = FLT_EPSILON * 4);
		static float degToRad(float degrees);
		static float radToDeg(float radians);
		static float smoothstep(float t);
		static float smootherstep(float t);
		static float fastPow(float a, float b);
	};
}
