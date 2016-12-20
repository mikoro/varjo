// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <string>

#include <Common.h>

namespace Varjo
{
	class CudaUtils
	{
	public:

		static void checkError(cudaError_t code, const std::string& message);
	};
}
