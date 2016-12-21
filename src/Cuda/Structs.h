// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <cfloat>

#include <cuda_runtime.h>

namespace Varjo
{
	struct CameraData
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
