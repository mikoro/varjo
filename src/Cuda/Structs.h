// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <cstdint>
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

	struct Intersection
	{
		float3 position;
		float3 normal;
		float2 texcoord;
		float distance = FLT_MAX;
		uint32_t triangleIndex = 0;
		uint32_t materialIndex = 0;
		bool wasFound = false;
	};

	// store state of the random generator
	struct Random
	{
		uint64_t state;
		uint64_t inc;
	};

	// store state of the sample generator
	struct Sample
	{
		uint32_t index;
		uint32_t permutation;
	};

	// path data in SOA format
	struct Paths
	{
		Random* random;
		Sample* filmSample;
		float2* filmSamplePosition;
		float3* throughput;
		float3* result;
		Ray* extensionRay;
		Ray* shadowRay;
		Intersection* intersection;
		bool* shadowRayBlocked;
		uint32_t* length;
	};

	// wavefront queues
	struct Queues
	{
		uint32_t* newPathQueue;
		uint32_t* materialQueue;
		uint32_t* extensionRayQueue;
		uint32_t* shadowRayQueue;

		uint32_t newPathQueueLength;
		uint32_t materialQueueLength;
		uint32_t extensionRayQueueLength;
		uint32_t shadowRayQueueLength;
	};

	// store film data in pixels
	struct Pixel
	{
		float3 color;
		float weight;
	};
}
