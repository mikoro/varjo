// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <cstdint>
#include <vector>
#include <string>

#include "Core/AABB.h"
#include "Core/Triangle.h"

namespace Varjo
{
	struct BVHNode
	{
		AABB aabb;
		int32_t rightOffset;
		uint32_t triangleOffset;
		uint32_t triangleCount;
	};

	class BVH
	{
	public:

		static void build(std::vector<Triangle>& triangles, std::vector<BVHNode>& nodes);
		static void exportDot(std::vector<BVHNode>& nodes, const std::string& fileName);
	};
}
