// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <cfloat>
#include <cstdint>

#include <cuda_runtime.h>

namespace Varjo
{
	namespace Cuda
	{
		struct Camera
		{
			float3 position;
			float3 right;
			float3 up;
			float3 forward;
			float3 filmCenter;
			float halfFilmWidth;
			float halfFilmHeight;
		};

		struct Ray
		{
			float3 origin;
			float3 direction;
			float3 invD;
			float3 OoD;
			float minDistance = 0.0f;
			float maxDistance = FLT_MAX;
		};

		struct Intersection
		{
			float3 position;
			float3 normal;
			float2 texcoord;
			float distance = FLT_MAX;
			bool wasFound = false;
		};

		struct AABB
		{
			float3 min;
			float3 max;
		};

		struct BVHNode
		{
			AABB aabb;
			int32_t rightOffset;
			uint32_t primitiveOffset;
			uint32_t primitiveCount;
			uint32_t splitAxis;
		};

		struct Sphere
		{
			float3 position;
			float radius;
		};

		struct Material
		{

		};

		struct Path
		{

		};
	}
}
