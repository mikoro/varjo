// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "Math/Vector2.h"

/*
 * http://graphics.pixar.com/library/MultiJitteredSampling/paper.pdf
 */

namespace Varjo
{
	class Sampler
	{
	public:

		static Vector2 getSample(uint32_t s, uint32_t m, uint32_t n, uint32_t p);
	};
}
