// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <cfloat>

#include <cuda_runtime.h>

namespace Varjo
{
	struct Ray
	{
		float3 origin;
		float3 direction;
		float3 invD;
		float3 OoD;
		float minDistance = 0.0f;
		float maxDistance = FLT_MAX;
	};
}
