// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <cstdint>
#include <vector>

#include "Core/AABB.h"
#include "Core/Sphere.h"

namespace Varjo
{
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

		static void build(std::vector<Sphere>& primitives, std::vector<BVHNode>& nodes);
		static void exportDot(std::vector<BVHNode>& nodes, const std::string& fileName);
	};
}
