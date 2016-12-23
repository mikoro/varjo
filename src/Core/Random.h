// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <cstdint>

namespace Varjo
{
	// http://www.pcg-random.org/
	struct RandomState
	{
		uint64_t state = 0x853c49e6748fea9bULL;
		uint64_t inc = 0xda3e39cb94b95bdbULL;
	};
}
