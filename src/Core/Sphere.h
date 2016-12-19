// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "Math/Vector3.h"

namespace Varjo
{
	class Ray;
	class Intersection;
	class AABB;

	class Sphere
	{
	public:

		CUDA_CALLABLE bool intersect(const Ray& ray, Intersection& intersection) const;
		AABB getAABB() const;
		
		Vector3 position;
		float radius = 1.0;
	};
}
