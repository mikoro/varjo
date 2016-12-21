// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <string>

#include <cuda_runtime.h>

namespace Varjo
{
	class CudaUtils
	{
	public:

		static void checkError(cudaError_t code, const std::string& message);
		static void initCuda();
		static void calculateDimensions(const void* kernel, const char* name, uint32_t width, uint32_t height, dim3& blockDim, dim3& gridDim);
	};
}
