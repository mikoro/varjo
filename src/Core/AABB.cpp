// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include <algorithm>
#include <cfloat>

#include "Core/AABB.h"
#include "Cuda/Math.h"
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

	min_.x = std::min(v0.x, std::min(v1.x, v2.x));
	min_.y = std::min(v0.y, std::min(v1.y, v2.y));
	min_.z = std::min(v0.z, std::min(v1.z, v2.z));

	float3 max_;

	max_.x = std::max(v0.x, std::max(v1.x, v2.x));
	max_.y = std::max(v0.y, std::max(v1.y, v2.y));
	max_.z = std::max(v0.z, std::max(v1.z, v2.z));

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
