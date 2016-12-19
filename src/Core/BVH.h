// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <cstdint>
#include <vector>

#include "Common.h"
#include "Core/AABB.h"
#include "Math/Vector3.h"

namespace Varjo
{
	class Sphere;
	class Ray;
	class Intersection;

	struct BVHNode
	{
		AABB aabb;
		int32_t rightOffset;
		uint32_t primitiveOffset;
		uint32_t primitiveCount;
		uint32_t splitAxis;
	};

	class BVH
	{
	public:

		static void build(std::vector<BVHNode>& nodes, std::vector<Sphere>& primitives);
		static CUDA_CALLABLE bool intersect(const Ray& ray, Intersection& intersection, const Sphere* primitives, const BVHNode* nodes);
	};
}
