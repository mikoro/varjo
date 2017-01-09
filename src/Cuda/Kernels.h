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
#include "Cuda/Material.h"

using namespace Varjo;

__global__ void initPathsKernel(Paths* paths, uint64_t seed, uint32_t pathCount)
{
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;

	if (id >= pathCount)
		return;

	initRandom(paths->random[id], seed, uint64_t(id));

	paths->filmSample[id].index = 0;
	paths->filmSample[id].permutation = randomInt(paths->random[id]);
	paths->result[id] = make_float3(0.0f, 0.0f, 0.0f);
	paths->length[id] = 0;
	paths->extensionIntersection[id] = Intersection();
}

__global__ void clearPathsKernel(Paths* paths, uint32_t pathCount, bool full)
{
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;

	if (id >= pathCount)
		return;

	if (full)
	{
		paths->length[id] = 0;
		paths->extensionIntersection[id] = Intersection();
	}
	else
	{
		if (paths->length[id] > 2)
			paths->extensionIntersection[id] = Intersection();
	}
}

__global__ void clearPixelsKernel(Pixel* pixels, uint32_t pixelCount)
{
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;

	if (id >= pixelCount)
		return;

	pixels[id].color = make_float3(0.0f, 0.0f, 0.0f);
	pixels[id].weight = 0.0f;
}

// https://mediatech.aalto.fi/~samuli/publications/laine2013hpg_paper.pdf
__global__ void logicKernel(
	Paths* __restrict paths,
	Queues* __restrict queues,
	const Triangle* __restrict triangles,
	const uint32_t* __restrict emitters,
	const Material* __restrict materials,
	Pixel* __restrict pixels,
	uint32_t pathCount,
	uint32_t emitterCount,
	uint32_t filmWidth,
	uint32_t filmHeight,
	uint32_t maxPathLength)
{
	const uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;

	if (id >= pathCount)
		return;

	bool terminate = false;
	float continueProb = 1.0f;
	Random random = paths->random[id];
	float3 throughput = paths->throughput[id];
	uint32_t pathLength = paths->length[id];

	// russian roulette
	if (pathLength > 2)
	{
		continueProb = clamp(fmaxf(throughput.x, fmaxf(throughput.y, throughput.z)), 0.0f, 0.95f);
		terminate = randomFloat(random) > continueProb;
	}

	if (pathLength > maxPathLength)
		terminate = true;

	// add direct emittance
	if (pathLength == 1 && paths->extensionIntersection[id].wasFound)
	{
		const Material& material = materials[paths->extensionIntersection[id].materialIndex];
		paths->result[id] += material.emittance;
	}

	// skip these for the first part of the path
	if (pathLength > 1)
	{
		// add direct light
		if (!paths->lightRayBlocked[id])
		{
			//float misPdf = 0.5f * paths->lightPdf[id] + 0.5f * paths->lightBrdfPdf[id];
			float misPdf = paths->lightPdf[id];

			if (misPdf > 0.0f)
				paths->result[id] += throughput * paths->lightEmittance[id] * paths->lightBrdf[id] * (paths->lightCosine[id] / misPdf);
		}

		float extensionBrdfPdf = paths->extensionBrdfPdf[id];

		// update path throughput
		if (extensionBrdfPdf > 0.0f)
			throughput *= paths->extensionBrdf[id] * (paths->extensionCosine[id] / extensionBrdfPdf / continueProb);
		else
			terminate = true;
	}

	// terminate path
	if (terminate || !paths->extensionIntersection[id].wasFound)
	{
		if (pathLength > 0)
			writeToPixels(paths, pixels, id, filmWidth, filmHeight);
		
		// add path to newPath queue
		uint32_t queueIndex = atomicAggInc(&queues->newPathQueueLength);
		queues->newPathQueue[queueIndex] = id;
	}
	else // path continues
	{
		generateLightSample(paths, triangles, emitters, materials, random, id, emitterCount);

		// add path to diffuse material queue
		uint32_t queueIndex = atomicAggInc(&queues->diffuseMaterialQueueLength);
		queues->diffuseMaterialQueue[queueIndex] = id;

		paths->throughput[id] = throughput;
	}

	paths->random[id] = random;
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

	// rotate sampling index and permutation
	if (paths->filmSample[id].index >= filmLength)
	{
		paths->filmSample[id].index = 0;
		paths->filmSample[id].permutation++;
	}

	float2 filmSamplePosition = getFilmSample(paths->filmSample[id].index++, filmWidth, filmHeight, paths->filmSample[id].permutation);
	paths->filmSamplePosition[id] = filmSamplePosition;
	paths->throughput[id] = make_float3(1.0f, 1.0f, 1.0f);
	paths->result[id] = make_float3(0.0f, 0.0f, 0.0f);
	paths->extensionRay[id] = getCameraRay(*camera, filmSamplePosition);
	paths->length[id] = 0;

	uint32_t queueIndex = atomicAggInc(&queues->extensionRayQueueLength);
	queues->extensionRayQueue[queueIndex] = id;
}

__global__ void diffuseMaterialKernel(
	Paths* __restrict paths,
	Queues* __restrict queues,
	const Material* __restrict materials)
{
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;

	if (id >= queues->diffuseMaterialQueueLength)
		return;

	id = queues->diffuseMaterialQueue[id];

	Random random = paths->random[id];
	const Intersection extensionIntersection = paths->extensionIntersection[id];
	const Material extensionMaterial = materials[extensionIntersection.materialIndex];

	float3 extensionDir = getDiffuseDirection(random, extensionIntersection.normal);

	paths->extensionBrdf[id] = getDiffuseBrdf(extensionMaterial);
	paths->extensionBrdfPdf[id] = getDiffusePdf(extensionIntersection.normal, extensionDir);
	paths->extensionCosine[id] = dot(extensionIntersection.normal, extensionDir);
	paths->lightBrdf[id] = getDiffuseBrdf(extensionMaterial);
	paths->lightBrdfPdf[id] = getDiffusePdf(extensionIntersection.normal, paths->lightRay[id].direction);

	Ray extensionRay;
	extensionRay.origin = extensionIntersection.position + RAY_EPSILON * extensionDir;
	extensionRay.direction = extensionDir;
	initRay(extensionRay);

	paths->extensionRay[id] = extensionRay;
	
	uint32_t queueIndex = atomicAggInc(&queues->extensionRayQueueLength);
	queues->extensionRayQueue[queueIndex] = id;

	queueIndex = atomicAggInc(&queues->lightRayQueueLength);
	queues->lightRayQueue[queueIndex] = id;

	paths->random[id] = random;
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

	// make normal always to face the ray
	if (intersection.wasFound && dot(intersection.normal, ray.direction) > 0.0f)
		intersection.normal = -intersection.normal;

	paths->length[id]++;
	paths->extensionIntersection[id] = intersection;
}

__global__ void lightRayKernel(
	Paths* __restrict paths,
	const Queues* __restrict queues,
	const BVHNode* __restrict nodes,
	const Triangle* __restrict triangles)
{
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;

	if (id >= queues->lightRayQueueLength)
		return;

	id = queues->lightRayQueue[id];
	
	Ray ray = paths->lightRay[id];
	Intersection intersection;
	intersectBvh(nodes, triangles, ray, intersection);
	paths->lightRayBlocked[id] = intersection.wasFound;
}

__global__ void writeFilmKernel(Pixel* pixels, uint32_t pixelCount, cudaSurfaceObject_t film, uint32_t filmWidth)
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

	// simple tonemapping
	const float invGamma = 1.0f / 2.2f;
	color = clamp(color, 0.0f, 1.0f);
	color.x = powf(color.x, invGamma);
	color.y = powf(color.y, invGamma);
	color.z = powf(color.z, invGamma);

	surf2Dwrite(make_float4(color, 1.0f), film, x * sizeof(float4), y, cudaBoundaryModeZero);
}
