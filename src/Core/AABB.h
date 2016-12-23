// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <cfloat>

#include <cuda_runtime.h>

namespace Varjo
{
	class AABB
	{
	public:

		static AABB createFromMinMax(const float3& min, const float3& max);
		static AABB createFromCenterExtent(const float3& center, const float3& extent);
		static AABB createFromVertices(const float3& v0, const float3& v1, const float3& v2);

		void expand(const AABB& other);

		float3 getCenter() const;
		float3 getExtent() const;
		float getSurfaceArea() const;

		float3 min = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
		float3 max = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
	};
}
