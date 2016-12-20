// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <cfloat>

#include <Common.h>

namespace Varjo
{
	class Intersection
	{
	public:

		float3 position;
		float3 normal;
		float2 texcoord;
		float distance = FLT_MAX;
		bool wasFound = false;
	};
}
