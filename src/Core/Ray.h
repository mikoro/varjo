// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <cfloat>

#include "Common.h"

namespace Varjo
{
	class Ray
	{
	public:

		float3 origin;
		float3 direction;
		float minDistance = 0.0f;
		float maxDistance = FLT_MAX;
	};
}
