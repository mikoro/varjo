// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "Cuda/Random.h"
#include "Cuda/Math.h"
#include "Cuda/Filtering.h"
#include "Cuda/Misc.h"
#include "Cuda/Sampling.h"
#include "Cuda/Camera.h"
#include "Cuda/Intersect.h"

using namespace Varjo;

__device__ const float RAY_EPSILON = 0.0001f;

__global__ void initPathsKernel(Paths* paths, uint64_t seed, uint32_t pathCount)
{
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;

	if (id >= pathCount)
		return;

	initRandom(paths->random[id], seed, uint64_t(id));

	paths->filmSample[id].index = 0;
	paths->filmSample[id].permutation = randomInt(paths->random[id]);
	paths->filmSamplePosition[id] = make_float2(0.0f, 0.0f);
	paths->throughput[id] = make_float3(0.0f, 0.0f, 0.0f);
	paths->result[id] = make_float3(0.0f, 0.0f, 0.0f);
	paths->extensionRay[id] = Ray();
	paths->shadowRay[id] = Ray();
	paths->intersection[id] = Intersection();
	paths->shadowRayBlocked[id] = true;
	paths->length[id] = 0;
}

__global__ void clearPixelsKernel(Pixel* pixels, uint32_t pixelCount)
{
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;

	if (id >= pixelCount)
		return;

	pixels[id].color = make_float3(0.0f, 0.0f, 0.0f);
	pixels[id].weight = 0.0f;
}

__global__ void writePixelsKernel(Pixel* pixels, uint32_t pixelCount, cudaSurfaceObject_t film, uint32_t filmWidth)
{
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;

	if (id >= pixelCount)
		return;

	uint32_t x = id % filmWidth;
	uint32_t y = id / filmWidth;
	float3 color = make_float3(0.0f, 0.0f, 0.0f);
	float weight = pixels[id].weight;

	// normalize pixel color value
	if (weight != 0.0f)
		color = pixels[id].color / weight;

	// simple tonemapping for the pixel color value
	const float invGamma = 1.0f / 2.2f;
	color = clamp(color, 0.0f, 1.0f);
	color.x = powf(color.x, invGamma);
	color.y = powf(color.y, invGamma);
	color.z = powf(color.z, invGamma);

	surf2Dwrite(make_float4(color, 1.0f), film, x * sizeof(float4), y, cudaBoundaryModeZero);
}

// https://mediatech.aalto.fi/~samuli/publications/laine2013hpg_paper.pdf
__global__ void logicKernel(
	Paths* __restrict paths,
	Queues* __restrict queues,
	const Triangle* __restrict triangles,
	const uint32_t* __restrict emitters,
	Pixel* __restrict pixels,
	uint32_t pathCount,
	uint32_t emitterCount,
	uint32_t filmWidth,
	uint32_t filmHeight)
{
	const uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;

	if (id >= pathCount)
		return;

	// russian roulette
	bool terminate = false;
	if (paths->length[id] > 2)
	{
		const float continueProb = 0.8f;
		terminate = randomFloat(paths->random[id]) > continueProb;
		paths->throughput[id] /= continueProb;
	}

	if (terminate || !paths->intersection[id].wasFound || isZero(paths->throughput[id])) // path is terminated
	{
		int ox = int(paths->filmSamplePosition[id].x);
		int oy = int(paths->filmSamplePosition[id].y);

		// add path result to film using mitchell filter
		for (int tx = -1; tx <= 2; ++tx)
		{
			for (int ty = -1; ty <= 2; ++ty)
			{
				int px = ox + tx;
				int py = oy + ty;
				px = clamp(px, 0, int(filmWidth) - 1);
				py = clamp(py, 0, int(filmHeight) - 1);
				float2 pixelPosition = make_float2(float(px), float(py));
				float2 distance = pixelPosition - paths->filmSamplePosition[id];
				float weight = mitchellFilter(distance.x) * mitchellFilter(distance.y);
				float3 color = weight * paths->result[id];
				int pixelIndex = py * int(filmWidth) + px;

				atomicAdd(&(pixels[pixelIndex].color.x), color.x);
				atomicAdd(&(pixels[pixelIndex].color.y), color.y);
				atomicAdd(&(pixels[pixelIndex].color.z), color.z);
				atomicAdd(&(pixels[pixelIndex].weight), weight);
			}
		}

		// add path to newPath queue
		uint32_t queueIndex = atomicAggInc(&queues->newPathQueueLength);
		queues->newPathQueue[queueIndex] = id;
	}
	else // path continues
	{
		// select random emitter triangle and generate shadow ray
		uint32_t emitterIndex = randomInt(paths->random[id], emitterCount);
		const Triangle& emitter = triangles[emitters[emitterIndex]];
		float3 emitterPosition = getTriangleSample(paths->random[id], emitter);
		float3 toEmitter = emitterPosition - paths->intersection[id].position;
		float3 toEmitterDir = normalize(toEmitter);
		float toEmitterDist = length(toEmitter);

		Ray shadowRay;
		shadowRay.origin = paths->intersection[id].position;
		shadowRay.direction = toEmitterDir;
		shadowRay.minDistance = RAY_EPSILON;
		shadowRay.maxDistance = toEmitterDist - RAY_EPSILON;
		initRay(shadowRay);

		paths->shadowRay[id] = shadowRay;

		// add path to material queue
		uint32_t queueIndex = atomicAggInc(&queues->materialQueueLength);
		queues->materialQueue[queueIndex] = id;
	}
}

__global__ void newPathKernel(
	Paths* __restrict paths,
	Queues* __restrict queues,
	const CameraData* __restrict camera,
	uint32_t filmWidth,
	uint32_t filmHeight,
	uint32_t filmLength)
{
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;

	if (id >= queues->newPathQueueLength)
		return;

	id = queues->newPathQueue[id];

	uint32_t filmSampleIndex = paths->filmSample[id].index;
	uint32_t filmSamplePermutation = paths->filmSample[id].permutation;

	// rotate sampling index and permutation
	if (filmSampleIndex >= filmLength)
	{
		filmSampleIndex = 0;
		paths->filmSample[id].permutation = ++filmSamplePermutation;
	}

	float2 filmSamplePosition = getFilmSample(filmSampleIndex++, filmWidth, filmHeight, filmSamplePermutation);
	paths->filmSample[id].index = filmSampleIndex;
	paths->filmSamplePosition[id] = filmSamplePosition;
	paths->throughput[id] = make_float3(1.0f, 1.0f, 1.0f);
	paths->result[id] = make_float3(0.0f, 0.0f, 0.0f);
	paths->extensionRay[id] = getCameraRay(*camera, filmSamplePosition);
	paths->shadowRay[id] = Ray();
	paths->intersection[id] = Intersection();
	paths->shadowRayBlocked[id] = true;
	paths->length[id] = 0;

	uint32_t queueIndex = atomicAggInc(&queues->extensionRayQueueLength);
	queues->extensionRayQueue[queueIndex] = id;
}

__global__ void materialKernel(
	Paths* __restrict paths,
	const Queues* __restrict queues,
	const Material* __restrict materials)
{
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;

	if (id >= queues->materialQueueLength)
		return;

	id = queues->materialQueue[id];

	const Intersection& intersection = paths->intersection[id];
	const Material& material = materials[intersection.materialIndex];
	paths->result[id] = material.baseColor * dot(paths->extensionRay[id].direction, -intersection.normal);
	paths->throughput[id] = make_float3(0.0f, 0.0f, 0.0f);
}

__global__ void extensionRayKernel(
	Paths* __restrict paths,
	const Queues* __restrict queues,
	const BVHNode* __restrict nodes,
	const Triangle* __restrict triangles)
{
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;

	if (id >= queues->extensionRayQueueLength)
		return;

	id = queues->extensionRayQueue[id];

	Ray ray = paths->extensionRay[id];
	Intersection intersection;
	intersectBvh(nodes, triangles, ray, intersection);
	paths->intersection[id] = intersection;
	paths->length[id]++;
}

__global__ void shadowRayKernel(
	Paths* __restrict paths,
	const Queues* __restrict queues,
	const BVHNode* __restrict nodes,
	const Triangle* __restrict triangles)
{
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;

	if (id >= queues->shadowRayQueueLength)
		return;

	id = queues->shadowRayQueue[id];
	
	Ray ray = paths->shadowRay[id];
	Intersection intersection;
	intersectBvh(nodes, triangles, ray, intersection);
	paths->shadowRayBlocked[id] = intersection.wasFound;
}
