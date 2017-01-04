// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "tinyformat/tinyformat.h"

#include "Utils/CudaUtils.h"

using namespace Varjo;

void CudaUtils::checkError(cudaError_t code, const std::string& message)
{
	if (code != cudaSuccess)
		throw std::runtime_error(tfm::format("CUDA error: %s: %s", message, cudaGetErrorString(code)));
}
