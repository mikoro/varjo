// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "Core/Random.h"
#include "Core/Sample.h"

namespace Varjo
{
	struct Path
	{
		Random random;
		Sample filmSample;
		Sample cameraSample;
		float2 filmSamplePosition;
		float3 result;
		float3 throughput;
	};
}
