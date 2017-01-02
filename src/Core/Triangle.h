// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <cstdint>

#include "Core/AABB.h"

namespace Varjo
{
	struct Triangle
	{
		void initialize();
		AABB getAABB() const;

		float3 vertices[3];
		float3 normals[3];
		float2 texcoords[3];
		float3 normal;
		float area;
		uint32_t materialIndex = 0;
	};
}
