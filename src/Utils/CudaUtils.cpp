// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include <cuda_runtime_api.h>

#include "tinyformat/tinyformat.h"

#include "Utils/CudaUtils.h"
#include "App.h"

using namespace Varjo;

void CudaUtils::checkError(cudaError_t code, const std::string& message)
{
	if (code != cudaSuccess)
		throw std::runtime_error(tfm::format("Cuda error: %s: %s", message, cudaGetErrorString(code)));
}
