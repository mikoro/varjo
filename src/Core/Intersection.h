// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <cstdint>
#include <cfloat>

#include <cuda_runtime.h>

namespace Varjo
{
	struct Intersection
	{
		float3 position;
		float3 normal;
		//float2 texcoord;
		float distance = FLT_MAX;
		uint32_t triangleIndex = 0;
		uint32_t materialIndex = 0;
		bool wasFound = false;
	};
}
