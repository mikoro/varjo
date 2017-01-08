// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "Utils/App.h"

using namespace Varjo;

inline void calculateDimensions(const void* kernel, const char* name, uint32_t length, int& blockSize, int& gridSize)
{
	int minGridSize;

	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel, 0, length);

	gridSize = (length + blockSize - 1) / blockSize;

	App::getLog().logInfo("Kernel (%s) block size: %d | grid size: %d", name, blockSize, gridSize);
}

inline void calculateDimensions2D(const void* kernel, const char* name, uint32_t width, uint32_t height, dim3& blockSize, dim3& gridSize)
{
	int tempBlockSize;
	int minGridSize;

	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &tempBlockSize, kernel, 0, width * height);

	blockSize.x = 32;
	blockSize.y = tempBlockSize / 32;

	if (blockSize.y == 0)
		blockSize.y = 1;

	gridSize.x = (width + blockSize.x - 1) / blockSize.x;
	gridSize.y = (height + blockSize.y - 1) / blockSize.y;

	App::getLog().logInfo("Kernel (%s) block size: %d (%dx%d) | grid size: %d (%dx%d)", name, tempBlockSize, blockSize.x, blockSize.y, gridSize.x * gridSize.y, gridSize.x, gridSize.y);
}

// https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics/
__device__ inline uint32_t atomicAggInc(uint32_t* ctr)
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

__device__ inline void initRay(Ray& ray)
{
	ray.invD = make_float3(1.0, 1.0f, 1.0f) / ray.direction;
	ray.OoD = ray.origin / ray.direction;
}
