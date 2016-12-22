// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include <cstdint>

#include <cuda/helper_math.h>
#include <device_launch_parameters.h>

#include "Cuda/Renderer.h"
#include "Cuda/Structs.h"
#include "Cuda/CudaUtils.h"
#include "Utils/App.h"

using namespace Varjo;
using namespace Cuda;

namespace
{
	__device__ void initRay(Ray& ray)
	{
		ray.invD = make_float3(1.0, 1.0f, 1.0f) / ray.direction;
		ray.OoD = ray.origin / ray.direction;
	}

	__device__ Ray getRay(float2 pointOnFilm, const Cuda::Camera& camera)
	{
		float dx = pointOnFilm.x - camera.halfFilmWidth;
		float dy = pointOnFilm.y - camera.halfFilmHeight;

		float3 positionOnFilm = camera.filmCenter + (dx * camera.right) + (dy * camera.up);

		Ray ray;
		ray.origin = camera.position;
		ray.direction = normalize(positionOnFilm - camera.position);
		initRay(ray);

		return ray;
	}

	__device__ void intersectSphere(const Cuda::Sphere& sphere, const Ray& ray, Intersection& intersection)
	{
		float3 rayOriginToSphere = sphere.position - ray.origin;
		float rayOriginToSphereDistance2 = dot(rayOriginToSphere, rayOriginToSphere);

		float t1 = dot(rayOriginToSphere, ray.direction);
		float sphereToRayDistance2 = rayOriginToSphereDistance2 - (t1 * t1);
		float radius2 = sphere.radius * sphere.radius;

		bool rayOriginIsOutside = (rayOriginToSphereDistance2 >= radius2);

		float t2 = sqrt(radius2 - sphereToRayDistance2);
		float t = (rayOriginIsOutside) ? (t1 - t2) : (t1 + t2);

		if (t1 > 0.0f && sphereToRayDistance2 < radius2 && t < intersection.distance)
		{
			intersection.wasFound = true;
			intersection.distance = t;
			intersection.position = ray.origin + (t * ray.direction);
			intersection.normal = normalize(intersection.position - sphere.position);
		}
	}

	__device__ bool intersectAabb(const Cuda::AABB& aabb, const Ray& ray)
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

	__device__ void intersectBvh(const Cuda::BVHNode* __restrict nodes, const Cuda::Sphere* __restrict primitives, const Ray& ray, Intersection& intersection)
	{
		uint32_t stack[16];
		uint32_t stackIndex = 1;
		stack[0] = 0;

		Cuda::BVHNode node;

		while (stackIndex > 0)
		{
			uint32_t nodeIndex = stack[--stackIndex];
			node = nodes[nodeIndex];

			// leaf node
			if (node.rightOffset == 0)
			{
				for (int i = 0; i < node.primitiveCount; ++i)
					intersectSphere(primitives[node.primitiveOffset + i], ray, intersection);

				continue;
			}

			if (intersectAabb(node.aabb, ray))
			{
				stack[stackIndex++] = nodeIndex + 1; // left child
				stack[stackIndex++] = nodeIndex + uint32_t(node.rightOffset); // right child
			}
		}
	}

	__global__ void clearKernel(cudaSurfaceObject_t film, uint32_t filmWidth, uint32_t filmHeight)
	{
		uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
		uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;

		if (x >= filmWidth || y >= filmHeight)
			return;

		float4 color = make_float4(1, 0, 0, 1);
		surf2Dwrite(color, film, x * sizeof(float4), y);
	}

	__global__ void traceKernel(const Cuda::Camera* __restrict camera, const Cuda::Sphere* __restrict primitives, const Cuda::BVHNode* __restrict nodes, cudaSurfaceObject_t film, uint32_t filmWidth, uint32_t filmHeight)
	{
		uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
		uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;

		Ray ray = getRay(make_float2(x, y), *camera);
		Intersection intersection;
		float4 color = make_float4(0.0f, 0.0f, 0.0f, 1.0f);

		intersectBvh(nodes, primitives, ray, intersection);

		if (intersection.wasFound)
			color = make_float4(1.0f, 0.0f, 0.0f, 1.0f) * dot(ray.direction, -intersection.normal);

		surf2Dwrite(color, film, x * sizeof(float4), y, cudaBoundaryModeZero);
	}
}

void Renderer::initialize(const Scene& scene)
{
	CudaUtils::checkError(cudaMallocManaged(&camera, sizeof(Cuda::Camera)), "Could not allocate CUDA device memory");
	CudaUtils::checkError(cudaMallocManaged(&primitives, sizeof(Sphere) * scene.primitives.size()), "Could not allocate CUDA device memory");
	CudaUtils::checkError(cudaMallocManaged(&nodes, sizeof(BVHNode) * scene.nodes.size()), "Could not allocate CUDA device memory");

	Cuda::Camera cameraData = scene.camera.getCudaCamera();

	memcpy(camera, &cameraData, sizeof(Cuda::Camera));
	memcpy(primitives, scene.primitives.data(), sizeof(Sphere) * scene.primitives.size());
	memcpy(nodes, scene.nodes.data(), sizeof(BVHNode) * scene.nodes.size());
}

void Renderer::shutdown()
{
	CudaUtils::checkError(cudaFree(primitives), "Could not free CUDA device memory");
	CudaUtils::checkError(cudaFree(nodes), "Could not free CUDA device memory");
	CudaUtils::checkError(cudaFree(camera), "Could not free CUDA device memory");
}

void Renderer::update(const Scene& scene)
{
	Cuda::Camera cameraData = scene.camera.getCudaCamera();
	memcpy(camera, &cameraData, sizeof(Cuda::Camera));
}

void Renderer::filmResized(uint32_t width, uint32_t height)
{
	CudaUtils::calculateDimensions(static_cast<void*>(clearKernel), "clear", width, height, clearKernelBlockDim, clearKernelGridDim);
	CudaUtils::calculateDimensions(static_cast<void*>(traceKernel), "trace", width, height, traceKernelBlockDim, traceKernelGridDim);
}

void Renderer::render()
{
	const Film& film = App::getWindow().getFilm();
	cudaGraphicsResource* filmTextureResource = film.getTextureResource();
	CudaUtils::checkError(cudaGraphicsMapResources(1, &filmTextureResource, 0), "Could not map CUDA texture resource");

	cudaArray_t filmTextureArray;
	CudaUtils::checkError(cudaGraphicsSubResourceGetMappedArray(&filmTextureArray, filmTextureResource, 0, 0), "Could not get CUDA mapped array");

	cudaResourceDesc filmResourceDesc;
	memset(&filmResourceDesc, 0, sizeof(filmResourceDesc));
	filmResourceDesc.resType = cudaResourceTypeArray;
	filmResourceDesc.res.array.array = filmTextureArray;

	cudaSurfaceObject_t filmSurfaceObject;
	CudaUtils::checkError(cudaCreateSurfaceObject(&filmSurfaceObject, &filmResourceDesc), "Could not create CUDA surface object");

	//clearKernel<<<clearKernelGridDim, clearKernelBlockDim>>>(filmSurfaceObject, film.getWidth(), film.getHeight());
	traceKernel<<<traceKernelGridDim, traceKernelBlockDim>>>(camera, primitives, nodes, filmSurfaceObject, film.getWidth(), film.getHeight());

	CudaUtils::checkError(cudaPeekAtLastError(), "Could not launch CUDA kernel");
	CudaUtils::checkError(cudaDeviceSynchronize(), "Could not execute CUDA kernel");

	CudaUtils::checkError(cudaDestroySurfaceObject(filmSurfaceObject), "Could not destroy CUDA surface object");
	CudaUtils::checkError(cudaGraphicsUnmapResources(1, &filmTextureResource, 0), "Could not unmap CUDA texture resource");
}
