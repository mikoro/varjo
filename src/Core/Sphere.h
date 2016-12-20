// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "Common.h"

namespace Varjo
{
	class Ray;
	class Intersection;
	class AABB;

	class Sphere
	{
	public:

		AABB getAABB() const;
		
		float3 position;
		float radius;
	};
}
