// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include <cstdint>

#include <cuda/helper_math.h>
#include <device_launch_parameters.h>

#include "Cuda/Renderer.h"
#include "Cuda/CudaUtils.h"
#include "Utils/App.h"
#include "Core/Ray.h"
#include "Core/Intersection.h"

using namespace Varjo;

namespace
{
	// http://www.pcg-random.org/
	__device__ uint32_t randomInt(Random& random)
	{
		uint64_t oldstate = random.state;
		random.state = oldstate * 6364136223846793005ULL + random.inc;
		uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
		uint32_t rot = oldstate >> 59u;
		return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
	}

	__device__ uint32_t randomInt(Random& random, uint32_t max)
	{
		uint32_t threshold = -max % max;

		for (;;)
		{
			uint32_t r = randomInt(random);

			if (r >= threshold)
				return r % max; // 0 <= r < max
		}
	}

	__device__ float randomFloat(Random& random)
	{
		//return float(ldexp(randomInt(random), -32)); // http://mumble.net/~campbell/tmp/random_real.c
		return float(randomInt(random)) / float(0xFFFFFFFF);
	}

	__device__ void initRandom(Random& random, uint64_t initstate, uint64_t initseq)
	{
		random.state = 0U;
		random.inc = (initseq << 1u) | 1u;
		randomInt(random);
		random.state += initstate;
		randomInt(random);
	}

	__device__ uint32_t permute(uint32_t i, uint32_t l, uint32_t p)
	{
		uint32_t w = l - 1;

		w |= w >> 1;
		w |= w >> 2;
		w |= w >> 4;
		w |= w >> 8;
		w |= w >> 16;

		do
		{
			i ^= p;
			i *= 0xe170893d;
			i ^= p >> 16;
			i ^= (i & w) >> 4;
			i ^= p >> 8;
			i *= 0x0929eb3f;
			i ^= p >> 23;
			i ^= (i & w) >> 1;
			i *= 1 | p >> 27;
			i *= 0x6935fa69;
			i ^= (i & w) >> 11;
			i *= 0x74dcb303;
			i ^= (i & w) >> 2;
			i *= 0x9e501cc3;
			i ^= (i & w) >> 2;
			i *= 0xc860a3df;
			i &= w;
			i ^= i >> 5;
		}
		while (i >= l);

		return (i + p) % l;
	}

	// http://graphics.pixar.com/library/MultiJitteredSampling/
	__device__ float2 getSample(uint32_t s, uint32_t m, uint32_t n, uint32_t p)
	{
		// if s is not permutated, the samples will come out in scanline order
		s = permute(s, m * n, p * 0xa511e9b3);

		uint32_t x = s % m;
		uint32_t y = s / m;

		uint32_t sx = permute(x, m, p * 0xa511e9b3);
		uint32_t sy = permute(y, n, p * 0x63d83595);

		float2 r;

		r.x = (float(x) + float(sy) / float(n)) / float(m);
		r.y = (float(y) + float(sx) / float(m)) / float(n);

		return r;
	}

	__device__ void initRay(Ray& ray)
	{
		ray.invD = make_float3(1.0, 1.0f, 1.0f) / ray.direction;
		ray.OoD = ray.origin / ray.direction;
	}

	__device__ Ray getRay(float2 filmPoint, const CameraData& camera)
	{
		float dx = filmPoint.x - camera.halfFilmWidth;
		float dy = filmPoint.y - camera.halfFilmHeight;

		float3 position = camera.filmCenter + (dx * camera.right) + (dy * camera.up);

		Ray ray;
		ray.origin = camera.position;
		ray.direction = normalize(position - camera.position);
		initRay(ray);

		return ray;
	}

	// http://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/moller-trumbore-ray-triangle-intersection
	__device__ void intersectTriangle(const Triangle* __restrict triangles, uint32_t triangleIndex, const Ray& ray, Intersection& intersection)
	{
		const Triangle triangle = triangles[triangleIndex];

		float3 v0v1 = triangle.vertices[1] - triangle.vertices[0];
		float3 v0v2 = triangle.vertices[2] - triangle.vertices[0];

		float3 pvec = cross(ray.direction, v0v2);
		float determinant = dot(v0v1, pvec);

		if (determinant == 0.0f)
			return;

		float invDeterminant = 1.0f / determinant;

		float3 tvec = ray.origin - triangle.vertices[0];
		float u = dot(tvec, pvec) * invDeterminant;

		if (u < 0.0f || u > 1.0f)
			return;

		float3 qvec = cross(tvec, v0v1);
		float v = dot(ray.direction, qvec) * invDeterminant;

		if (v < 0.0f || (u + v) > 1.0f)
			return;

		float distance = dot(v0v2, qvec) * invDeterminant;

		if (distance < 0.0f || distance < ray.minDistance || distance > ray.maxDistance || distance > intersection.distance)
			return;

		float w = 1.0f - u - v;

		intersection.wasFound = true;
		intersection.distance = distance;
		intersection.position = ray.origin + (distance * ray.direction);
		intersection.normal = w * triangle.normals[0] + u * triangle.normals[1] + v * triangle.normals[2];
		//intersection.texcoord = w * triangle.texcoords[0] + u * triangle.texcoords[1] + v * triangle.texcoords[2];
		intersection.triangleIndex = triangleIndex;
		intersection.materialIndex = triangle.materialIndex;
	}

	// https://mediatech.aalto.fi/~timo/publications/aila2012hpg_techrep.pdf
	__device__ bool intersectAabb(const AABB& aabb, const Ray& ray)
	{
		float x0 = fmaf(aabb.min.x, ray.invD.x, -ray.OoD.x);
		float y0 = fmaf(aabb.min.y, ray.invD.y, -ray.OoD.y);
		float z0 = fmaf(aabb.min.z, ray.invD.z, -ray.OoD.z);
		float x1 = fmaf(aabb.max.x, ray.invD.x, -ray.OoD.x);
		float y1 = fmaf(aabb.max.y, ray.invD.y, -ray.OoD.y);
		float z1 = fmaf(aabb.max.z, ray.invD.z, -ray.OoD.z);

		float tminbox = fmaxf(fmaxf(ray.minDistance, fminf(x0, x1)), fmaxf(fminf(y0, y1), fminf(z0, z1)));
		float tmaxbox = fminf(fminf(ray.maxDistance, fmaxf(x0, x1)), fminf(fmaxf(y0, y1), fmaxf(z0, z1)));

		return (tminbox <= tmaxbox);
	}

	__device__ void intersectBvh(const BVHNode* __restrict nodes, const Triangle* __restrict triangles, const Ray& ray, Intersection& intersection)
	{
		BVHNode node;
		uint32_t stack[64];
		uint32_t stackIndex = 1;
		stack[0] = 0;

		while (stackIndex > 0)
		{
			uint32_t nodeIndex = stack[--stackIndex];
			node = nodes[nodeIndex];

			// leaf node
			if (node.rightOffset == 0)
			{
				for (int i = 0; i < node.triangleCount; ++i)
					intersectTriangle(triangles, node.triangleOffset + i, ray, intersection);

				continue;
			}

			if (intersectAabb(node.aabb, ray))
			{
				stack[stackIndex++] = nodeIndex + 1; // left child
				stack[stackIndex++] = nodeIndex + uint32_t(node.rightOffset); // right child
			}
		}
	}

	__global__ void initPathsKernel(Path* paths, uint64_t seed, uint32_t pathCount)
	{
		uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;

		if (id >= pathCount)
			return;

		initRandom(paths[id].random, seed, uint64_t(id));

		paths[id].filmSample.index = 0;
		paths[id].filmSample.permutation = randomInt(paths[id].random);
	}

	__global__ void clearPixelsKernel(Pixel* pixels, uint32_t pixelCount)
	{
		uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;

		if (id >= pixelCount)
			return;

		pixels[id].value = make_float3(0.0f, 0.0f, 0.0f);
		pixels[id].weight = 0.0f;
	}

	__global__ void writePixelsKernel(Pixel* pixels, uint32_t pixelCount, cudaSurfaceObject_t film, uint32_t filmWidth)
	{
		uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;

		if (id >= pixelCount)
			return;

		uint32_t x = id % filmWidth;
		uint32_t y = id / filmWidth;
		float3 result = make_float3(0.0f, 0.0f, 0.0f);
		float weight = pixels[id].weight;

		if (weight != 0.0f)
			result = pixels[id].value / weight;

		surf2Dwrite(make_float4(result, 1.0f), film, x * sizeof(float4), y, cudaBoundaryModeZero);
	}

	/*
	 uint32_t filmSampleIndex = paths[id].filmSample.index;
	 uint32_t filmSamplePermutation = paths[id].filmSample.permutation;

	 if (filmSampleIndex >= filmLength)
	 {
	 filmSampleIndex = 0;
	 paths[id].filmSample.permutation = ++filmSamplePermutation;
	 }

	 float2 filmSamplePosition = getSample(filmSampleIndex++, filmWidth, filmHeight, filmSamplePermutation);
	 paths[id].filmSample.index = filmSampleIndex;
	 paths[id].filmSamplePosition = filmSamplePosition;
	 paths[id].ray = getRay(filmSamplePosition, *camera);
	 paths[id].result = make_float3(0.0f, 0.0f, 0.0f);
	 paths[id].throughput = make_float3(1.0f, 1.0f, 1.0f);
	 paths[id].state == PathState::RAY_CONTINUATION;
	 */

	__global__ void logicKernel(Path* __restrict paths, uint32_t pathCount, uint32_t filmWidth, uint32_t filmHeight, uint32_t filmLength)
	{
		const uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;

		if (id >= pathCount)
			return;
	}

	__global__ void newPathKernel(const uint32_t* __restrict newPathQueue, uint32_t newPathQueueLength, Path* __restrict paths, const CameraData* __restrict camera)
	{
		const uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;

		if (id >= newPathQueueLength)
			return;
	}

	__global__ void materialKernel(const uint32_t* __restrict materialQueue, uint32_t materialQueueLength, Path* __restrict paths, const Material* __restrict materials)
	{
		const uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;

		if (id >= materialQueueLength)
			return;

		/*Intersection intersection = paths[id].continuationIntersection;
		const Material material = materials[intersection.materialIndex];
		paths[id].result = material.baseColor * dot(paths[id].ray.direction, -intersection.normal);*/
	}

	__global__ void extensionRayKernel(
		const uint32_t* __restrict extensionRayQueue,
		uint32_t extensionRayQueueLength,
		Path* __restrict paths,
		const BVHNode* __restrict nodes,
		const Triangle* __restrict triangles)
	{
		const uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;

		if (id >= extensionRayQueueLength)
			return;

		//Intersection intersection;
		//intersectBvh(nodes, triangles, ray, intersection);
	}

	__global__ void shadowRayKernel(
		const uint32_t* __restrict shadowRayQueue,
		uint32_t shadowRayQueueLength,
		Path* __restrict paths,
		const BVHNode* __restrict nodes,
		const Triangle* __restrict triangles)
	{
		const uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;

		if (id >= shadowRayQueueLength)
			return;

		//Intersection intersection;
		//intersectBvh(nodes, triangles, ray, intersection);
	}
}

void Renderer::initialize(const Scene& scene)
{
	CudaUtils::checkError(cudaMallocManaged(&camera, sizeof(CameraData)), "Could not allocate CUDA device memory");
	CudaUtils::checkError(cudaMallocManaged(&nodes, sizeof(BVHNode) * scene.nodes.size()), "Could not allocate CUDA device memory");
	CudaUtils::checkError(cudaMallocManaged(&triangles, sizeof(Triangle) * scene.triangles.size()), "Could not allocate CUDA device memory");
	CudaUtils::checkError(cudaMallocManaged(&emitters, sizeof(Triangle) * scene.emitters.size()), "Could not allocate CUDA device memory");
	CudaUtils::checkError(cudaMallocManaged(&materials, sizeof(Material) * scene.materials.size()), "Could not allocate CUDA device memory");
	CudaUtils::checkError(cudaMallocManaged(&paths, sizeof(Path) * pathCount), "Could not allocate CUDA device memory");
	CudaUtils::checkError(cudaMallocManaged(&queues, sizeof(Queues)), "Could not allocate CUDA device memory");
	CudaUtils::checkError(cudaMallocManaged(&queues->newPathQueue, sizeof(uint32_t) * pathCount), "Could not allocate CUDA device memory");
	CudaUtils::checkError(cudaMallocManaged(&queues->materialQueue, sizeof(uint32_t) * pathCount), "Could not allocate CUDA device memory");
	CudaUtils::checkError(cudaMallocManaged(&queues->extensionRayQueue, sizeof(uint32_t) * pathCount), "Could not allocate CUDA device memory");
	CudaUtils::checkError(cudaMallocManaged(&queues->shadowRayQueue, sizeof(uint32_t) * pathCount), "Could not allocate CUDA device memory");

	CameraData cameraData = scene.camera.getCameraData();

	memcpy(camera, &cameraData, sizeof(CameraData));
	memcpy(nodes, scene.nodes.data(), sizeof(BVHNode) * scene.nodes.size());
	memcpy(triangles, scene.triangles.data(), sizeof(Triangle) * scene.triangles.size());
	memcpy(emitters, scene.emitters.data(), sizeof(Triangle) * scene.emitters.size());
	memcpy(materials, scene.materials.data(), sizeof(Material) * scene.materials.size());

	uint64_t time = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
	int blockSize, gridSize;
	CudaUtils::calculateDimensions(static_cast<void*>(initPathsKernel), "initPaths", pathCount, blockSize, gridSize);
	initPathsKernel<<<gridSize, blockSize>>>(paths, time, pathCount);
	CudaUtils::checkError(cudaPeekAtLastError(), "Could not launch CUDA kernel");
	CudaUtils::checkError(cudaDeviceSynchronize(), "Could not execute CUDA kernel");

	queues->newPathQueueLength = 0;
	queues->materialQueueLength = 0;
	queues->extensionRayQueueLength = 0;
	queues->shadowRayQueueLength = 0;

	CudaUtils::calculateDimensions(static_cast<void*>(logicKernel), "logicKernel", pathCount, logicBlockSize, logicGridSize);
	CudaUtils::calculateDimensions(static_cast<void*>(newPathKernel), "newPathKernel", pathCount, newPathBlockSize, newPathGridSize);
	CudaUtils::calculateDimensions(static_cast<void*>(materialKernel), "materialKernel", pathCount, materialBlockSize, materialGridSize);
	CudaUtils::calculateDimensions(static_cast<void*>(extensionRayKernel), "extensionRayKernel", pathCount, extensionRayBlockSize, extensionRayGridSize);
	CudaUtils::calculateDimensions(static_cast<void*>(shadowRayKernel), "shadowRayKernel", pathCount, shadowRayBlockSize, shadowRayGridSize);
}

void Renderer::shutdown()
{
	CudaUtils::checkError(cudaFree(camera), "Could not free CUDA device memory");
	CudaUtils::checkError(cudaFree(nodes), "Could not free CUDA device memory");
	CudaUtils::checkError(cudaFree(triangles), "Could not free CUDA device memory");
	CudaUtils::checkError(cudaFree(emitters), "Could not free CUDA device memory");
	CudaUtils::checkError(cudaFree(materials), "Could not free CUDA device memory");
	CudaUtils::checkError(cudaFree(paths), "Could not free CUDA device memory");
	CudaUtils::checkError(cudaFree(queues->newPathQueue), "Could not free CUDA device memory");
	CudaUtils::checkError(cudaFree(queues->materialQueue), "Could not free CUDA device memory");
	CudaUtils::checkError(cudaFree(queues->extensionRayQueue), "Could not free CUDA device memory");
	CudaUtils::checkError(cudaFree(queues->shadowRayQueue), "Could not free CUDA device memory");
	CudaUtils::checkError(cudaFree(queues), "Could not free CUDA device memory");
}

void Renderer::update(const Scene& scene)
{
	CameraData cameraData = scene.camera.getCameraData();
	memcpy(camera, &cameraData, sizeof(CameraData));
}

void Renderer::filmResized(uint32_t filmWidth, uint32_t filmHeight)
{
	if (pixels != nullptr)
		CudaUtils::checkError(cudaFree(pixels), "Could not free CUDA device memory");

	pixelCount = filmWidth * filmHeight;
	CudaUtils::checkError(cudaMallocManaged(&pixels, sizeof(Pixel) * pixelCount), "Could not allocate CUDA device memory");
	CudaUtils::calculateDimensions(static_cast<void*>(clearPixelsKernel), "clearPixelsKernel", pixelCount, clearPixelsBlockSize, clearPixelsGridSize);
	CudaUtils::calculateDimensions(static_cast<void*>(writePixelsKernel), "writePixelsKernel", pixelCount, writePixelsBlockSize, writePixelsGridSize);

	clear();
}

void Renderer::clear()
{
	clearPixelsKernel<<<clearPixelsGridSize, clearPixelsBlockSize>>>(pixels, pixelCount);
	CudaUtils::checkError(cudaPeekAtLastError(), "Could not launch CUDA kernel");
	CudaUtils::checkError(cudaDeviceSynchronize(), "Could not execute CUDA kernel");
}

void Renderer::render()
{
	Film& film = App::getWindow().getFilm();

	logicKernel<<<logicGridSize, logicBlockSize>>>(paths, pathCount, film.getWidth(), film.getHeight(), film.getLength());
	CudaUtils::checkError(cudaPeekAtLastError(), "Could not launch CUDA kernel");
	CudaUtils::checkError(cudaDeviceSynchronize(), "Could not execute CUDA kernel");

	if (queues->newPathQueueLength > 0)
	{
		newPathGridSize = (queues->newPathQueueLength + newPathBlockSize - 1) / newPathBlockSize;
		newPathKernel<<<newPathGridSize, newPathBlockSize>>>(queues->newPathQueue, queues->newPathQueueLength, paths, camera);
		CudaUtils::checkError(cudaPeekAtLastError(), "Could not launch CUDA kernel");
		CudaUtils::checkError(cudaDeviceSynchronize(), "Could not execute CUDA kernel");
		queues->newPathQueueLength = 0;
	}

	if (queues->materialQueueLength > 0)
	{
		materialGridSize = (queues->materialQueueLength + materialBlockSize - 1) / materialBlockSize;
		materialKernel<<<materialGridSize, materialBlockSize>>>(queues->materialQueue, queues->materialQueueLength, paths, materials);
		CudaUtils::checkError(cudaPeekAtLastError(), "Could not launch CUDA kernel");
		CudaUtils::checkError(cudaDeviceSynchronize(), "Could not execute CUDA kernel");
		queues->materialQueueLength = 0;
	}

	if (queues->extensionRayQueueLength > 0)
	{
		extensionRayGridSize = (queues->extensionRayQueueLength + extensionRayBlockSize - 1) / extensionRayBlockSize;
		extensionRayKernel<<<extensionRayGridSize, extensionRayBlockSize>>>(queues->extensionRayQueue, queues->extensionRayQueueLength, paths, nodes, triangles);
		CudaUtils::checkError(cudaPeekAtLastError(), "Could not launch CUDA kernel");
		CudaUtils::checkError(cudaDeviceSynchronize(), "Could not execute CUDA kernel");
		queues->extensionRayQueueLength = 0;
	}

	if (queues->shadowRayQueueLength > 0)
	{
		shadowRayGridSize = (queues->shadowRayQueueLength + shadowRayBlockSize - 1) / shadowRayBlockSize;
		shadowRayKernel<<<shadowRayGridSize, shadowRayBlockSize>>>(queues->shadowRayQueue, queues->shadowRayQueueLength, paths, nodes, triangles);
		CudaUtils::checkError(cudaPeekAtLastError(), "Could not launch CUDA kernel");
		CudaUtils::checkError(cudaDeviceSynchronize(), "Could not execute CUDA kernel");
		queues->shadowRayQueueLength = 0;
	}

	cudaSurfaceObject_t filmSurfaceObject = film.getFilmSurfaceObject();
	writePixelsKernel<<<writePixelsGridSize, writePixelsBlockSize>>>(pixels, pixelCount, filmSurfaceObject, film.getWidth());
	CudaUtils::checkError(cudaPeekAtLastError(), "Could not launch CUDA kernel");
	CudaUtils::checkError(cudaDeviceSynchronize(), "Could not execute CUDA kernel");
	film.releaseFilmSurfaceObject();
}
