// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include <cstdint>
#include <device_launch_parameters.h>

#include "Renderer.h"
#include "Utils/CudaUtils.h"
#include "Utils/App.h"

using namespace Varjo;

namespace
{
	__global__ void updateFilmTexture(cudaSurfaceObject_t output, uint32_t filmWidth, uint32_t filmHeight)
	{
		uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
		uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;

		if (x >= filmWidth || y >= filmHeight)
			return;

		float4 color = make_float4(1, 0, 0, 1);
		surf2Dwrite(color, output, x * sizeof(float4), y);
	}
}

void Renderer::initialize()
{
	
}

void Renderer::shutdown()
{
	
}

void Renderer::render()
{
	const Film& film = App::getWindow().getFilm();
	cudaGraphicsResource* texture = film.getTexture();
	CudaUtils::checkError(cudaGraphicsMapResources(1, &texture, 0), "Could not map CUDA texture resource");

	cudaArray_t textureData;
	CudaUtils::checkError(cudaGraphicsSubResourceGetMappedArray(&textureData, texture, 0, 0), "Could not get CUDA mapped array");

	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = textureData;

	cudaSurfaceObject_t output;
	CudaUtils::checkError(cudaCreateSurfaceObject(&output, &resDesc), "Could not create CUDA surface object");

	dim3 dimBlock(16, 16);
	dim3 dimGrid;

	dimGrid.x = (film.getWidth() + dimBlock.x - 1) / dimBlock.x;
	dimGrid.y = (film.getHeight() + dimBlock.y - 1) / dimBlock.y;

	updateFilmTexture<<<dimGrid, dimBlock>>>(output, film.getWidth(), film.getHeight());

	CudaUtils::checkError(cudaPeekAtLastError(), "Could not launch update film texture kernel");
	CudaUtils::checkError(cudaDeviceSynchronize(), "Could not execute update film texture kernel");

	CudaUtils::checkError(cudaDestroySurfaceObject(output), "Could not destroy CUDA surface object");
	CudaUtils::checkError(cudaGraphicsUnmapResources(1, &texture, 0), "Could not unmap CUDA texture resource");
}
