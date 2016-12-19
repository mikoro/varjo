// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include <cstdint>
#include <device_launch_parameters.h>

#include "Core/Renderer.h"
#include "Core/Intersection.h"
#include "Utils/CudaUtils.h"
#include "Utils/App.h"

using namespace Varjo;

namespace
{
	__global__ void clearFilm(cudaSurfaceObject_t film, uint32_t filmWidth, uint32_t filmHeight)
	{
		uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
		uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;

		if (x >= filmWidth || y >= filmHeight)
			return;

		float4 color = make_float4(1, 0, 0, 1);
		surf2Dwrite(color, film, x * sizeof(float4), y);
	}

	__global__ void trace(const Sphere* primitives, const BVHNode* nodes, const Camera* camera, cudaSurfaceObject_t film, uint32_t filmWidth, uint32_t filmHeight)
	{
		uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
		uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;

		if (x >= filmWidth || y >= filmHeight)
			return;

		Ray ray = camera->getRay(Vector2(float(x), float(y)));
		Intersection intersection;

		Color color = Color::black();

		//if(BVH::intersect(ray, intersection, primitives, nodes))
		//{
		//	//color = Color::red() * (ray.direction.dot(intersection.normal));
		//	color = Color::red();
		//}

		for (int i = 0; i < 121; ++i)
			primitives[i].intersect(ray, intersection);

		if (intersection.wasFound)
		{
			//color = Color::red() * (ray.direction.dot(intersection.normal));
			color = Color::red();
		}

		surf2Dwrite(make_float4(color.r, color.g, color.b, 1.0f), film, x * sizeof(float4), y);
	}
}

void Renderer::initialize(const Scene& scene)
{
	CudaUtils::checkError(cudaMalloc(&primitives, sizeof(Sphere) * scene.primitives.size()), "Could not allocate CUDA device memory");
	CudaUtils::checkError(cudaMalloc(&nodes, sizeof(BVHNode) * scene.nodes.size()), "Could not allocate CUDA device memory");
	CudaUtils::checkError(cudaMalloc(&camera, sizeof(Camera)), "Could not allocate CUDA device memory");

	CudaUtils::checkError(cudaMemcpy(primitives, scene.primitives.data(), sizeof(Sphere) * scene.primitives.size(), cudaMemcpyHostToDevice), "Could not write data to CUDA device");
	CudaUtils::checkError(cudaMemcpy(nodes, scene.nodes.data(), sizeof(BVHNode) * scene.nodes.size(), cudaMemcpyHostToDevice), "Could not write data to CUDA device");
	CudaUtils::checkError(cudaMemcpy(camera, &scene.camera, sizeof(Camera), cudaMemcpyHostToDevice), "Could not write data to CUDA device");
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

	dim3 dimBlock(16, 16);
	dim3 dimGrid;

	dimGrid.x = (film.getWidth() + dimBlock.x - 1) / dimBlock.x;
	dimGrid.y = (film.getHeight() + dimBlock.y - 1) / dimBlock.y;

	trace<<<dimGrid, dimBlock>>>(primitives, nodes, camera, filmSurfaceObject, film.getWidth(), film.getHeight());

	CudaUtils::checkError(cudaPeekAtLastError(), "Could not launch CUDA kernel");
	CudaUtils::checkError(cudaDeviceSynchronize(), "Could not execute CUDA kernel");

	CudaUtils::checkError(cudaDestroySurfaceObject(filmSurfaceObject), "Could not destroy CUDA surface object");
	CudaUtils::checkError(cudaGraphicsUnmapResources(1, &filmTextureResource, 0), "Could not unmap CUDA texture resource");
}
