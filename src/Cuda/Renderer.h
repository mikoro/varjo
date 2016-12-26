// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <cuda_runtime.h>

#include "Core/Scene.h"
#include "Core/Camera.h"
#include "Core/BVH.h"
#include "Core/Triangle.h"
#include "Core/Material.h"
#include "Core/Ray.h"
#include "Core/Intersection.h"

namespace Varjo
{
	struct Random
	{
		uint64_t state;
		uint64_t inc;
	};

	struct Sample
	{
		uint32_t index;
		uint32_t permutation;
	};

	struct Path
	{
		Random random;
		Sample filmSample;
		float2 filmSamplePosition;
		Ray extensionRay;
		Ray shadowRay;
		Intersection extensionIntersection;
		Intersection shadowIntersection;
		float3 result;
		float3 throughput;
	};

	struct Queues
	{
		uint32_t newPathQueueLength;
		uint32_t* newPathQueue;
		uint32_t materialQueueLength;
		uint32_t* materialQueue;
		uint32_t extensionRayQueueLength;
		uint32_t* extensionRayQueue;
		uint32_t shadowRayQueueLength;
		uint32_t* shadowRayQueue;
	};

	struct Pixel
	{
		float3 value;
		float weight;
	};

	class Renderer
	{
	public:

		void initialize(const Scene& scene);
		void shutdown();
		void update(const Scene& scene);
		void filmResized(uint32_t filmWidth, uint32_t filmHeight);
		void clear();
		void render();

	private:

		const uint32_t pathCount = 256000;

		CameraData* camera = nullptr;
		BVHNode* nodes = nullptr;
		Triangle* triangles = nullptr;
		Triangle* emitters = nullptr;
		Material* materials = nullptr;
		Path* paths = nullptr;
		Queues* queues = nullptr;
		Pixel* pixels = nullptr;
		
		uint32_t pixelCount;
		int clearPixelsBlockSize;
		int clearPixelsGridSize;
		int writePixelsBlockSize;
		int writePixelsGridSize;
		int logicBlockSize;
		int logicGridSize;
		int newPathBlockSize;
		int newPathGridSize;
		int materialBlockSize;
		int materialGridSize;
		int extensionRayBlockSize;
		int extensionRayGridSize;
		int shadowRayBlockSize;
		int shadowRayGridSize;
	};
}
