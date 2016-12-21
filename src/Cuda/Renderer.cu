// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include <cstdint>

#include <cuda/helper_math.h>
#include <device_launch_parameters.h>

#include "Cuda/Renderer.h"
#include "Cuda/Structs.h"
#include "Utils/CudaUtils.h"
#include "Utils/App.h"

using namespace Varjo;

namespace
{
	void calculateDimensions(const void* kernel, const char* name, uint32_t width, uint32_t height, dim3& blockDim, dim3& gridDim)
	{
		int blockSize;
		int minGridSize;

		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel, 0, width * height);

		assert(blockSize % 32 == 0);

		blockDim.x = 32;
		blockDim.y = blockSize / 32;

		gridDim.x = (width + blockDim.x - 1) / blockDim.x;
		gridDim.y = (height + blockDim.y - 1) / blockDim.y;

		App::getLog().logInfo("Kernel (%s) block size: %d (%dx%d) | grid size: %d (%dx%d)", name, blockSize, blockDim.x, blockDim.y, gridDim.x * gridDim.y, gridDim.x, gridDim.y);
	}

	__device__ Ray getRay(float2 pointOnFilm, const CameraData& camera)
	{
		float dx = pointOnFilm.x - camera.halfFilmWidth;
		float dy = pointOnFilm.y - camera.halfFilmHeight;

		float3 positionOnFilm = camera.filmCenter + (dx * camera.right) + (dy * camera.up);

		Ray ray;
		ray.origin = camera.position;
		ray.direction = normalize(positionOnFilm - camera.position);

		return ray;
	}

	__device__ void intersectSphere(const Sphere& sphere, const Ray& ray, Intersection& intersection)
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

	__global__ void clearKernel(cudaSurfaceObject_t film, uint32_t filmWidth, uint32_t filmHeight)
	{
		uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
		uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;

		if (x >= filmWidth || y >= filmHeight)
			return;

		float4 color = make_float4(1, 0, 0, 1);
		surf2Dwrite(color, film, x * sizeof(float4), y);
	}

	__global__ void traceKernel(const CameraData* __restrict cameraPtr, const Sphere* __restrict primitives, cudaSurfaceObject_t film, uint32_t filmWidth, uint32_t filmHeight)
	{
		uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
		uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;

		const CameraData& camera = *cameraPtr;
		Ray ray = getRay(make_float2(x, y), camera);
		Intersection intersection;
		float4 color = make_float4(0.0f, 0.0f, 0.0f, 1.0f);

		for (int i = 0; i < 121; ++i)
		{
			intersectSphere(primitives[i], ray, intersection);
		}



		if (intersection.wasFound)
			color = make_float4(1.0f, 0.0f, 0.0f, 1.0f) * dot(ray.direction, -intersection.normal);

		surf2Dwrite(color, film, x * sizeof(float4), y, cudaBoundaryModeZero);
	}
}

void Renderer::initialize(const Scene& scene)
{
	CudaUtils::checkError(cudaMallocManaged(&primitives, sizeof(Sphere) * scene.primitives.size()), "Could not allocate CUDA device memory");
	CudaUtils::checkError(cudaMallocManaged(&nodes, sizeof(BVHNode) * scene.nodes.size()), "Could not allocate CUDA device memory");
	CudaUtils::checkError(cudaMallocManaged(&camera, sizeof(CameraData)), "Could not allocate CUDA device memory");

	CameraData cameraData = scene.camera.getCameraData();

	memcpy(primitives, scene.primitives.data(), sizeof(Sphere) * scene.primitives.size());
	memcpy(nodes, scene.nodes.data(), sizeof(BVHNode) * scene.nodes.size());
	memcpy(camera, &cameraData, sizeof(CameraData));

	const Film& film = App::getWindow().getFilm();

	calculateDimensions(static_cast<void*>(clearKernel), "clear", film.getWidth(), film.getHeight(), clearKernelBlockDim, clearKernelGridDim);
	calculateDimensions(static_cast<void*>(traceKernel), "trace", film.getWidth(), film.getHeight(), traceKernelBlockDim, traceKernelGridDim);
}

void Renderer::shutdown()
{
	CudaUtils::checkError(cudaFree(primitives), "Could not free CUDA device memory");
	CudaUtils::checkError(cudaFree(nodes), "Could not free CUDA device memory");
	CudaUtils::checkError(cudaFree(camera), "Could not free CUDA device memory");
}

void Renderer::update(const Scene& scene)
{
	CameraData cameraData = scene.camera.getCameraData();

	memcpy(camera, &cameraData, sizeof(CameraData));
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
	traceKernel<<<traceKernelGridDim, traceKernelBlockDim>>>(camera, primitives, filmSurfaceObject, film.getWidth(), film.getHeight());

	CudaUtils::checkError(cudaPeekAtLastError(), "Could not launch CUDA kernel");
	CudaUtils::checkError(cudaDeviceSynchronize(), "Could not execute CUDA kernel");

	CudaUtils::checkError(cudaDestroySurfaceObject(filmSurfaceObject), "Could not destroy CUDA surface object");
	CudaUtils::checkError(cudaGraphicsUnmapResources(1, &filmTextureResource, 0), "Could not unmap CUDA texture resource");
}
