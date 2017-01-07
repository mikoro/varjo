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
	paths->length[id] = 0;
	paths->extensionRay[id] = Ray();
	paths->extensionIntersection[id] = Intersection();
	paths->extensionBrdf[id] = make_float3(0.0f, 0.0f, 0.0f);
	paths->extensionBrdfPdf[id] = 0.0f;
	paths->extensionCosine[id] = 0.0f;
	paths->lightRay[id] = Ray();
	paths->lightIntersection[id] = Intersection();
	paths->lightEmittance[id] = make_float3(0.0f, 0.0f, 0.0f);
	paths->lightBrdf[id] = make_float3(0.0f, 0.0f, 0.0f);
	paths->lightBrdfPdf[id] = 0.0f;
	paths->lightPdf[id] = 0.0f;
	paths->lightCosine[id] = 0.0f;
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

	// simple tonemapping
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

	bool terminate = false;
	float continueProb = 1.0f;

	// russian roulette
	if (paths->length[id] > 2)
	{
		continueProb = 0.8f;
		terminate = randomFloat(paths->random[id]) > continueProb;
	}

	// terminate too long paths
	if (paths->length[id] > 8)
		terminate = true;

	// skip these for the first part of the path
	if (paths->length[id] > 0)
	{
		// add direct light
		if (!paths->lightIntersection[id].wasFound)
			paths->result[id] += paths->throughput[id] * paths->lightEmittance[id] * paths->lightBrdf[id] * (paths->lightCosine[id] / paths->lightPdf[id]);

		// update path throughput
		paths->throughput[id] *= paths->extensionBrdf[id] * (paths->extensionCosine[id] / paths->extensionBrdfPdf[id] / continueProb);
	}

	paths->length[id]++;

	// terminate path
	if (terminate || !paths->extensionIntersection[id].wasFound)
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
		// select random emitter triangle and generate light ray
		Ray lightRay;
		uint32_t triangleId = emitters[randomInt(paths->random[id], emitterCount)];
		const Triangle& triangle = triangles[triangleId];
		Intersection triangleIntersection = getTriangleSample(paths->random[id], triangle);
		float3 toTriangle = triangleIntersection.position - paths->extensionIntersection[id].position;
		float3 toTriangleDir = normalize(toTriangle);
		float toTriangleDist = length(toTriangle);
		lightRay.origin = paths->extensionIntersection[id].position;
		lightRay.direction = toTriangleDir;
		lightRay.minDistance = RAY_EPSILON;
		lightRay.maxDistance = toTriangleDist - RAY_EPSILON;
		initRay(lightRay);
		triangleIntersection.triangleIndex = triangleId;
		triangleIntersection.distance = toTriangleDist;
		paths->lightRay[id] = lightRay;
		paths->lightIntersection[id] = triangleIntersection;

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

__global__ void materialKernel(
	Paths* __restrict paths,
	Queues* __restrict queues,
	const Material* __restrict materials)
{
	uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;

	if (id >= queues->materialQueueLength)
		return;

	id = queues->materialQueue[id];

	const Intersection& extensionIntersection = paths->extensionIntersection[id];
	const Intersection& lightIntersection = paths->lightIntersection[id];
	const Material& extensionMaterial = materials[extensionIntersection.materialIndex];
	const Material& lightMaterial = materials[lightIntersection.materialIndex];

	float3 extensionDir = getDiffuseDirection(paths->random[id], extensionIntersection.normal);
	float3 extensionBrdf = getDiffuseBrdf(extensionMaterial);
	float extensionBrdfPdf = getDiffusePdf(extensionIntersection.normal, extensionDir);
	float extensionCosine = dot(extensionIntersection.normal, extensionDir);

	float3 lightEmittance = lightMaterial.emittance;
	float3 lightBrdf = getDiffuseBrdf(extensionMaterial);
	float lightBrdfPdf = getDiffusePdf(extensionIntersection.normal, paths->lightRay[id].direction);
	float lightPdf = (lightIntersection.distance * lightIntersection.distance) / dot(lightIntersection.normal, -paths->lightRay[id].direction);
	float lightCosine = dot(extensionIntersection.normal, paths->lightRay[id].direction);

	Ray extensionRay;
	extensionRay.origin = extensionIntersection.position;
	extensionRay.direction = extensionDir;
	initRay(extensionRay);

	paths->extensionRay[id] = extensionRay;
	paths->extensionBrdf[id] = extensionBrdf;
	paths->extensionBrdfPdf[id] = extensionBrdfPdf;
	paths->extensionCosine[id] = extensionCosine;

	paths->lightEmittance[id] = lightEmittance;
	paths->lightBrdf[id] = lightBrdf;
	paths->lightBrdfPdf[id] = lightBrdfPdf;
	paths->lightPdf[id] = lightPdf;
	paths->lightCosine[id] = lightCosine;

	uint32_t queueIndex = atomicAggInc(&queues->extensionRayQueueLength);
	queues->extensionRayQueue[queueIndex] = id;

	queueIndex = atomicAggInc(&queues->lightRayQueueLength);
	queues->lightRayQueue[queueIndex] = id;
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
	paths->lightIntersection[id].wasFound = intersection.wasFound;
}
