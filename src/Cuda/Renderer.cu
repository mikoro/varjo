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

__device__ const float RAY_EPSILON = 0.0001f;

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
	__device__ float2 getFilmSample(uint32_t s, uint32_t m, uint32_t n, uint32_t p)
	{
		// if s is not permutated, the samples will come out in scanline order
		s = permute(s, m * n, p * 0xa511e9b3);

		uint32_t x = s % m;
		uint32_t y = s / m;

		uint32_t sx = permute(x, m, p * 0xa511e9b3);
		uint32_t sy = permute(y, n, p * 0x63d83595);

		float2 r;

		r.x = float(x) + float(sy) / float(n);
		r.y = float(y) + float(sx) / float(m);

		return r;
	}

	__device__ float3 getCosineSample(Random& random, const float3& normal)
	{
		float u1 = randomFloat(random);
		float u2 = randomFloat(random);
		float r = sqrtf(u1);
		float theta = 2.0f * float(M_PI) * u2;
		float x = r * cosf(theta);
		float y = r * sinf(theta);
		float z = sqrtf(fmaxf(0.0f, 1.0f - u1));

		float3 u = normalize(cross(normal, make_float3(0.0001f, 1.0000f, 0.0001f)));
		float3 v = normalize(cross(u, normal));

		return x * u + y * v + z * normal;
	}

	__device__ float getCosinePdf(const float3& normal, const float3& direction)
	{
		return dot(normal, direction) / float(M_PI);
	}

	__device__ float3 getTriangleSample(Random& random, const Triangle& triangle)
	{
		float r1 = randomFloat(random);
		float r2 = randomFloat(random);
		float sr1 = sqrtf(r1);

		float u = 1.0f - sr1;
		float v = r2 * sr1;
		float w = 1.0f - u - v;

		return u * triangle.vertices[0] + v * triangle.vertices[1] + w * triangle.vertices[2];
	}

	__device__ float mitchellFilter(float s)
	{
		const float B = 1.0f / 3.0f;
		const float C = 1.0f / 3.0f;

		s = abs(s);

		if (s < 1.0f)
			return ((12.0f - 9.0f * B - 6.0f * C) * (s * s * s) + (-18.0f + 12.0f * B + 6.0f * C) * (s * s) + (6.0f - 2.0f * B)) * (1.0f / 6.0f);

		if (s < 2.0f)
			return ((-B - 6.0f * C) * (s * s * s) + (6.0f * B + 30.0f * C) * (s * s) + (-12.0f * B - 48.0f * C) * s + (8.0f * B + 24.0f * C)) * (1.0f / 6.0f);

		return 0.0f;
	}

	__device__ void initRay(Ray& ray)
	{
		ray.invD = make_float3(1.0, 1.0f, 1.0f) / ray.direction;
		ray.OoD = ray.origin / ray.direction;
	}

	__device__ Ray getCameraRay(const CameraData& camera, float2 filmPoint)
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
		const Triangle& triangle = triangles[triangleIndex];

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
		uint32_t stack[16];
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

		initRandom(paths->random[id], seed, uint64_t(id));

		paths->filmSample[id].index = 0;
		paths->filmSample[id].permutation = randomInt(paths->random[id]);
		paths->filmSamplePosition[id] = make_float2(0.0f, 0.0f);
		paths->throughput[id] = make_float3(0.0f, 0.0f, 0.0f);
		paths->color[id] = make_float3(0.0f, 0.0f, 0.0f);
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

		if (weight != 0.0f)
			color = pixels[id].color / weight;

		const float invGamma = 1.0f / 2.2f;
		color = clamp(color, 0.0f, 1.0f);
		color.x = powf(color.x, invGamma);
		color.y = powf(color.y, invGamma);
		color.z = powf(color.z, invGamma);
		
		surf2Dwrite(make_float4(color, 1.0f), film, x * sizeof(float4), y, cudaBoundaryModeZero);
	}

	// https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics/
	__device__ uint32_t atomicAggInc(uint32_t* ctr)
	{
		uint32_t mask = __ballot(1);
		uint32_t leader = __ffs(mask) - 1;
		uint32_t laneid = threadIdx.x % 32;
		uint32_t res;

		if (laneid == leader)
			res = atomicAdd(ctr, __popc(mask));

		res = __shfl(res, leader);
		return res + __popc(mask & ((1 << laneid) - 1));
	}

	// https://mediatech.aalto.fi/~samuli/publications/laine2013hpg_paper.pdf
	__global__ void logicKernel(
		Path* __restrict paths,
		Queues* __restrict queues,
		uint32_t* __restrict emitters,
		Pixel* __restrict pixels,
		uint32_t pathCount,
		uint32_t emitterCount,
		uint32_t filmWidth,
		uint32_t filmHeight)
	{
		const uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;

		if (id >= pathCount)
			return;

		// add russian roulette

		if (!paths->intersection[id].wasFound || isZero(paths->throughput[id])) // path is terminated
		{
			int ox = int(paths->filmSamplePosition[id].x);
			int oy = int(paths->filmSamplePosition[id].y);

			for (int tx = -1; tx <= 2; ++tx)
			{
				for (int ty = -1; ty <= 2; ++ty)
				{
					int px = ox + tx;
					int py = oy + ty;
					px = clamp(px, 0, int(filmWidth));
					py = clamp(py, 0, int(filmHeight));
					float2 pixelPosition = make_float2(float(px), float(py));
					float2 distance = pixelPosition - paths->filmSamplePosition[id];
					float weight = mitchellFilter(distance.x) * mitchellFilter(distance.y);
					float3 color = weight * paths->color[id];
					int pixelIndex = py * int(filmWidth) + px;
					
					atomicAdd(&(pixels[pixelIndex].color.x), color.x);
					atomicAdd(&(pixels[pixelIndex].color.y), color.y);
					atomicAdd(&(pixels[pixelIndex].color.z), color.z);
					atomicAdd(&(pixels[pixelIndex].weight), weight);
				}
			}

			uint32_t queueIndex = atomicAggInc(&queues->newPathQueueLength);
			queues->newPathQueue[queueIndex] = id;
		}
		else // determine intersection material, add to material queue
		{
			uint32_t queueIndex = atomicAggInc(&queues->materialQueueLength);
			queues->materialQueue[queueIndex] = id;
		}
	}

	__global__ void newPathKernel(
		Path* __restrict paths,
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

		if (filmSampleIndex >= filmLength)
		{
			filmSampleIndex = 0;
			paths->filmSample[id].permutation = ++filmSamplePermutation;
		}

		float2 filmSamplePosition = getFilmSample(filmSampleIndex++, filmWidth, filmHeight, filmSamplePermutation);
		paths->filmSample[id].index = filmSampleIndex;
		paths->filmSamplePosition[id] = filmSamplePosition;
		paths->throughput[id] = make_float3(1.0f, 1.0f, 1.0f);
		paths->color[id] = make_float3(0.0f, 0.0f, 0.0f);
		paths->ray[id] = getCameraRay(*camera, filmSamplePosition);

		uint32_t queueIndex = atomicAggInc(&queues->extensionRayQueueLength);
		queues->extensionRayQueue[queueIndex] = id;
	}

	__global__ void materialKernel(
		Path* __restrict paths,
		const Queues* __restrict queues,
		const Material* __restrict materials)
	{
		uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;

		if (id >= queues->materialQueueLength)
			return;

		id = queues->materialQueue[id];

		const Intersection& intersection = paths->intersection[id];
		const Material& material = materials[intersection.materialIndex];
		paths->color[id] = material.baseColor * dot(paths->ray[id].direction, -intersection.normal);
		paths->throughput[id] = make_float3(0.0f, 0.0f, 0.0f);
	}

	__global__ void extensionRayKernel(
		Path* __restrict paths,
		const Queues* __restrict queues,
		const BVHNode* __restrict nodes,
		const Triangle* __restrict triangles)
	{
		uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;

		if (id >= queues->extensionRayQueueLength)
			return;

		id = queues->extensionRayQueue[id];

		Ray ray = paths->ray[id];
		Intersection intersection;
		intersectBvh(nodes, triangles, ray, intersection);
		paths->intersection[id] = intersection;
	}

	__global__ void directLightKernel(
		Path* __restrict paths,
		const Queues* __restrict queues,
		const BVHNode* __restrict nodes,
		const Triangle* __restrict triangles,
		const uint32_t* __restrict emitters,
		uint32_t emitterCount)
	{
		uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;

		if (id >= queues->directLightQueueLength)
			return;

		id = queues->directLightQueue[id];
		uint32_t emitterIndex = randomInt(paths->random[id], emitterCount);
		Triangle emitter = triangles[emitters[emitterIndex]];
		float3 emitterPosition = getTriangleSample(paths->random[id], emitter);
		float3 toEmitter = emitterPosition - paths->intersection[id].position;
		float3 toEmitterDir = normalize(toEmitter);
		float toEmitterDist = length(toEmitter);

		Ray ray;
		ray.origin = paths->intersection[id].position;
		ray.direction = toEmitterDir;
		ray.minDistance = RAY_EPSILON;
		ray.maxDistance = toEmitterDist - RAY_EPSILON;
		initRay(ray);

		Intersection intersection;
		intersectBvh(nodes, triangles, ray, intersection);

		if (!intersection.wasFound)
		{
			
		}
	}
}

void Renderer::initialize(const Scene& scene)
{
	CudaUtils::checkError(cudaMallocManaged(&camera, sizeof(CameraData)), "Could not allocate CUDA device memory");
	CudaUtils::checkError(cudaMallocManaged(&nodes, sizeof(BVHNode) * scene.nodes.size()), "Could not allocate CUDA device memory");
	CudaUtils::checkError(cudaMallocManaged(&triangles, sizeof(Triangle) * scene.triangles.size()), "Could not allocate CUDA device memory");
	CudaUtils::checkError(cudaMallocManaged(&emitters, sizeof(uint32_t) * scene.emitters.size()), "Could not allocate CUDA device memory");
	CudaUtils::checkError(cudaMallocManaged(&materials, sizeof(Material) * scene.materials.size()), "Could not allocate CUDA device memory");
	CudaUtils::checkError(cudaMallocManaged(&paths, sizeof(Path)), "Could not allocate CUDA device memory");
	CudaUtils::checkError(cudaMallocManaged(&paths->random, sizeof(Random) * pathCount), "Could not allocate CUDA device memory");
	CudaUtils::checkError(cudaMallocManaged(&paths->filmSample, sizeof(Sample) * pathCount), "Could not allocate CUDA device memory");
	CudaUtils::checkError(cudaMallocManaged(&paths->filmSamplePosition, sizeof(float2) * pathCount), "Could not allocate CUDA device memory");
	CudaUtils::checkError(cudaMallocManaged(&paths->throughput, sizeof(float3) * pathCount), "Could not allocate CUDA device memory");
	CudaUtils::checkError(cudaMallocManaged(&paths->color, sizeof(float3) * pathCount), "Could not allocate CUDA device memory");
	CudaUtils::checkError(cudaMallocManaged(&paths->ray, sizeof(Ray) * pathCount), "Could not allocate CUDA device memory");
	CudaUtils::checkError(cudaMallocManaged(&paths->intersection, sizeof(Intersection) * pathCount), "Could not allocate CUDA device memory");
	CudaUtils::checkError(cudaMallocManaged(&queues, sizeof(Queues)), "Could not allocate CUDA device memory");
	CudaUtils::checkError(cudaMallocManaged(&queues->newPathQueue, sizeof(uint32_t) * pathCount), "Could not allocate CUDA device memory");
	CudaUtils::checkError(cudaMallocManaged(&queues->materialQueue, sizeof(uint32_t) * pathCount), "Could not allocate CUDA device memory");
	CudaUtils::checkError(cudaMallocManaged(&queues->extensionRayQueue, sizeof(uint32_t) * pathCount), "Could not allocate CUDA device memory");
	CudaUtils::checkError(cudaMallocManaged(&queues->directLightQueue, sizeof(uint32_t) * pathCount), "Could not allocate CUDA device memory");

	CameraData cameraData = scene.camera.getCameraData();

	memcpy(camera, &cameraData, sizeof(CameraData));
	memcpy(nodes, scene.nodes.data(), sizeof(BVHNode) * scene.nodes.size());
	memcpy(triangles, scene.triangles.data(), sizeof(Triangle) * scene.triangles.size());
	memcpy(emitters, scene.emitters.data(), sizeof(uint32_t) * scene.emitters.size());
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
	queues->directLightQueueLength = 0;

	CudaUtils::calculateDimensions(static_cast<void*>(logicKernel), "logicKernel", pathCount, logicBlockSize, logicGridSize);
	CudaUtils::calculateDimensions(static_cast<void*>(newPathKernel), "newPathKernel", pathCount, newPathBlockSize, newPathGridSize);
	CudaUtils::calculateDimensions(static_cast<void*>(materialKernel), "materialKernel", pathCount, materialBlockSize, materialGridSize);
	CudaUtils::calculateDimensions(static_cast<void*>(extensionRayKernel), "extensionRayKernel", pathCount, extensionRayBlockSize, extensionRayGridSize);
	CudaUtils::calculateDimensions(static_cast<void*>(directLightKernel), "directLightKernel", pathCount, directLightBlockSize, directLightGridSize);

	averagePathsPerSecond.setAlpha(0.05f);
	averageRaysPerSecond.setAlpha(0.05f);

	emitterCount = uint32_t(scene.emitters.size());
}

void Renderer::shutdown()
{
	CudaUtils::checkError(cudaFree(camera), "Could not free CUDA device memory");
	CudaUtils::checkError(cudaFree(nodes), "Could not free CUDA device memory");
	CudaUtils::checkError(cudaFree(triangles), "Could not free CUDA device memory");
	CudaUtils::checkError(cudaFree(emitters), "Could not free CUDA device memory");
	CudaUtils::checkError(cudaFree(materials), "Could not free CUDA device memory");
	CudaUtils::checkError(cudaFree(paths->random), "Could not free CUDA device memory");
	CudaUtils::checkError(cudaFree(paths->filmSample), "Could not free CUDA device memory");
	CudaUtils::checkError(cudaFree(paths->filmSamplePosition), "Could not free CUDA device memory");
	CudaUtils::checkError(cudaFree(paths->throughput), "Could not free CUDA device memory");
	CudaUtils::checkError(cudaFree(paths->color), "Could not free CUDA device memory");
	CudaUtils::checkError(cudaFree(paths->ray), "Could not free CUDA device memory");
	CudaUtils::checkError(cudaFree(paths->intersection), "Could not free CUDA device memory");
	CudaUtils::checkError(cudaFree(paths), "Could not free CUDA device memory");
	CudaUtils::checkError(cudaFree(queues->newPathQueue), "Could not free CUDA device memory");
	CudaUtils::checkError(cudaFree(queues->materialQueue), "Could not free CUDA device memory");
	CudaUtils::checkError(cudaFree(queues->extensionRayQueue), "Could not free CUDA device memory");
	CudaUtils::checkError(cudaFree(queues->directLightQueue), "Could not free CUDA device memory");
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

	logicKernel<<<logicGridSize, logicBlockSize>>>(paths, queues, emitters, pixels, pathCount, emitterCount, film.getWidth(), film.getHeight());
	newPathKernel<<<newPathGridSize, newPathBlockSize>>>(paths, queues, camera, film.getWidth(), film.getHeight(), film.getLength());
	materialKernel<<<materialGridSize, materialBlockSize>>>(paths, queues, materials);
	extensionRayKernel<<<extensionRayGridSize, extensionRayBlockSize>>>(paths, queues, nodes, triangles);

	cudaSurfaceObject_t filmSurfaceObject = film.getFilmSurfaceObject();
	writePixelsKernel<<<writePixelsGridSize, writePixelsBlockSize>>>(pixels, pixelCount, filmSurfaceObject, film.getWidth());
	CudaUtils::checkError(cudaPeekAtLastError(), "Could not launch CUDA kernel");
	CudaUtils::checkError(cudaDeviceSynchronize(), "Could not execute CUDA kernel");
	film.releaseFilmSurfaceObject();

	float elapsedSeconds = timer.getElapsedSeconds();
	averagePathsPerSecond.addMeasurement(float(queues->newPathQueueLength) / elapsedSeconds);
	averageRaysPerSecond.addMeasurement(float(queues->extensionRayQueueLength + queues->directLightQueueLength) / elapsedSeconds);
	timer.restart();

	queues->newPathQueueLength = 0;
	queues->materialQueueLength = 0;
	queues->extensionRayQueueLength = 0;
	queues->directLightQueueLength = 0;
}

float Renderer::getPathsPerSecond() const
{
	return averagePathsPerSecond.getAverage();
}

float Renderer::getRaysPerSecond() const
{
	return averageRaysPerSecond.getAverage();
}
