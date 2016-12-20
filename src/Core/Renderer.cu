// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include <cstdint>
#include <cuda_runtime.h>
#include <helper_math.h>

#include "Core/Renderer.h"
#include "Core/Intersection.h"
#include "Utils/CudaUtils.h"
#include "Utils/App.h"

using namespace Varjo;

namespace
{
	void calculateDimensions(const void* kernel, const char* name, const Film& film, dim3& blockDim, dim3& gridDim)
	{
		int blockSize;
		int minGridSize;

		cudaOccupancyMaxPotentialBlockSize(
			&minGridSize,
			&blockSize,
			kernel,
			0,
			film.getLength());

		assert(blockSize % 32 == 0);

		blockDim.x = 32;
		blockDim.y = blockSize / 32;

		gridDim.x = (film.getWidth() + blockDim.x - 1) / blockDim.x;
		gridDim.y = (film.getHeight() + blockDim.y - 1) / blockDim.y;

		App::getLog().logInfo("Kernel (%s) block size: %d (%dx%d) | grid size: %d (%dx%d)", name, blockSize, blockDim.x, blockDim.y, gridDim.x * gridDim.y, gridDim.x, gridDim.y);
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

	__global__ void traceKernel(cudaSurfaceObject_t film, uint32_t filmWidth, uint32_t filmHeight)
	{
		uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
		uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;

		float3 cameraPosition = make_float3(0.0f, 0.0f, 10.0f);

		float dx = float(x) - float(filmWidth) / 2.0f;
		float dy = float(y) - float(filmHeight) / 2.0f;

		float3 positionOnFilm = make_float3(0.0f, 0.0f, -407.032135f) + (dx * make_float3(1.0f, 0.0f, 0.0f)) + (dy * make_float3(0.0f, 1.0f, 0.0f));
		float3 rayDirection = normalize(positionOnFilm - cameraPosition);

		bool wasFound = false;
		float3 intersectionPosition;
		float3 intersectionNormal;
		float intersectionDistance = FLT_MAX;
		float4 color = make_float4(0.0f, 0.0f, 0.0f, 1.0f);

		for (int sy = -10; sy <= 10; sy += 2)
		{
			for (int sx = -10; sx <= 10; sx += 2)
			{
				float3 spherePosition = make_float3(sx, sy, 0.0f);
				float3 rayOriginToSphere = spherePosition - cameraPosition;
				float rayOriginToSphereDistance2 = dot(rayOriginToSphere, rayOriginToSphere);
				float t1 = dot(rayOriginToSphere, rayDirection);
				float sphereToRayDistance2 = rayOriginToSphereDistance2 - (t1 * t1);
				float radius2 = 1.0f * 1.0f;
				bool rayOriginIsOutside = (rayOriginToSphereDistance2 >= radius2);

				float t2 = sqrt(radius2 - sphereToRayDistance2);
				float t = (rayOriginIsOutside) ? (t1 - t2) : (t1 + t2);

				if (t1 > 0.0f && sphereToRayDistance2 < radius2 && t < intersectionDistance)
				{
					wasFound = true;
					intersectionDistance = t;
					intersectionPosition = cameraPosition + (t * rayDirection);
					intersectionNormal = normalize(intersectionPosition - spherePosition);
				}

				//intersection.texcoord.x = 0.5f + atan2(normal.z, normal.x) / (2.0f * float(M_PI));
				//intersection.texcoord.y = 0.5f - asin(normal.y) / float(M_PI);
			}
		}

		if (wasFound)
			color = make_float4(1.0f, 0.0f, 0.0f, 1.0f) * dot(rayDirection, -intersectionNormal);

		//if (wasFound)
		//	color = make_float4(1.0f, 0.0f, 0.0f, 1.0f);

		surf2Dwrite(color, film, x * sizeof(float4), y, cudaBoundaryModeZero);
	}
}

void Renderer::initialize(const Scene& scene)
{
	const Film& film = App::getWindow().getFilm();

	CudaUtils::checkError(cudaMalloc(&primitives, sizeof(Sphere) * scene.primitives.size()), "Could not allocate CUDA device memory");
	CudaUtils::checkError(cudaMalloc(&nodes, sizeof(BVHNode) * scene.nodes.size()), "Could not allocate CUDA device memory");
	CudaUtils::checkError(cudaMalloc(&camera, sizeof(Camera)), "Could not allocate CUDA device memory");

	CudaUtils::checkError(cudaMemcpy(primitives, scene.primitives.data(), sizeof(Sphere) * scene.primitives.size(), cudaMemcpyHostToDevice), "Could not write data to CUDA device");
	CudaUtils::checkError(cudaMemcpy(nodes, scene.nodes.data(), sizeof(BVHNode) * scene.nodes.size(), cudaMemcpyHostToDevice), "Could not write data to CUDA device");
	CudaUtils::checkError(cudaMemcpy(camera, &scene.camera, sizeof(Camera), cudaMemcpyHostToDevice), "Could not write data to CUDA device");

	calculateDimensions(static_cast<void*>(clearKernel), "clear", film, clearKernelBlockDim, clearKernelGridDim);
	calculateDimensions(static_cast<void*>(traceKernel), "trace", film, traceKernelBlockDim, traceKernelGridDim);
}

void Renderer::shutdown()
{
	CudaUtils::checkError(cudaFree(primitives), "Could not free CUDA device memory");
	CudaUtils::checkError(cudaFree(nodes), "Could not free CUDA device memory");
	CudaUtils::checkError(cudaFree(camera), "Could not free CUDA device memory");
}

void Renderer::update(const Scene& scene)
{
	CudaUtils::checkError(cudaMemcpy(camera, &scene.camera, sizeof(Camera), cudaMemcpyHostToDevice), "Could not write data to CUDA device");
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
	traceKernel<<<traceKernelGridDim, traceKernelBlockDim>>>(filmSurfaceObject, film.getWidth(), film.getHeight());

	CudaUtils::checkError(cudaPeekAtLastError(), "Could not launch CUDA kernel");
	CudaUtils::checkError(cudaDeviceSynchronize(), "Could not execute CUDA kernel");

	CudaUtils::checkError(cudaDestroySurfaceObject(filmSurfaceObject), "Could not destroy CUDA surface object");
	CudaUtils::checkError(cudaGraphicsUnmapResources(1, &filmTextureResource, 0), "Could not unmap CUDA texture resource");
}
