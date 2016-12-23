// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include <cfloat>

#include <cuda/helper_math.h>

#include "Core/AABB.h"
#include "Common.h"

using namespace Varjo;

AABB AABB::createFromMinMax(const float3& min_, const float3& max_)
{
	AABB aabb;

	aabb.min = min_;
	aabb.max = max_;

	return aabb;
}

AABB AABB::createFromCenterExtent(const float3& center, const float3& extent)
{
	AABB aabb;

	aabb.min = center - extent / 2.0f;
	aabb.max = center + extent / 2.0f;

	return aabb;
}

AABB AABB::createFromVertices(const float3& v0, const float3& v1, const float3& v2)
{
	float3 min_;

	min_.x = MIN(v0.x, MIN(v1.x, v2.x));
	min_.y = MIN(v0.y, MIN(v1.y, v2.y));
	min_.z = MIN(v0.z, MIN(v1.z, v2.z));

	float3 max_;

	max_.x = MAX(v0.x, MAX(v1.x, v2.x));
	max_.y = MAX(v0.y, MAX(v1.y, v2.y));
	max_.z = MAX(v0.z, MAX(v1.z, v2.z));

	return AABB::createFromMinMax(min_, max_);
}

void AABB::expand(const AABB& other)
{
	if (other.min.x < min.x)
		min.x = other.min.x;

	if (other.min.y < min.y)
		min.y = other.min.y;

	if (other.min.z < min.z)
		min.z = other.min.z;

	if (other.max.x > max.x)
		max.x = other.max.x;

	if (other.max.y > max.y)
		max.y = other.max.y;

	if (other.max.z > max.z)
		max.z = other.max.z;
}

float3 AABB::getCenter() const
{
	return (min + max) * 0.5;
}

float3 AABB::getExtent() const
{
	return max - min;
}

float AABB::getSurfaceArea() const
{
	float3 extent = getExtent();
	return 2.0f * (extent.x * extent.y + extent.z * extent.y + extent.x * extent.z);
}
