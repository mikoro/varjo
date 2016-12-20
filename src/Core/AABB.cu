// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include <cfloat>

#include "Core/AABB.h"
#include "Core/Ray.h"

using namespace Varjo;

AABB::AABB()
{
	min.x = min.y = min.z = FLT_MAX;
	max.x = max.y = max.z = -FLT_MAX;
}

AABB AABB::createFromMinMax(const Vector3& min_, const Vector3& max_)
{
	AABB aabb;

	aabb.min = min_;
	aabb.max = max_;

	return aabb;
}

AABB AABB::createFromCenterExtent(const Vector3& center, const Vector3& extent)
{
	AABB aabb;

	aabb.min = center - extent / 2.0f;
	aabb.max = center + extent / 2.0f;

	return aabb;
}

AABB AABB::createFromVertices(const Vector3& v0, const Vector3& v1, const Vector3& v2)
{
	Vector3 min_;

	min_.x = MIN(v0.x, MIN(v1.x, v2.x));
	min_.y = MIN(v0.y, MIN(v1.y, v2.y));
	min_.z = MIN(v0.z, MIN(v1.z, v2.z));

	Vector3 max_;

	max_.x = MAX(v0.x, MAX(v1.x, v2.x));
	max_.y = MAX(v0.y, MAX(v1.y, v2.y));
	max_.z = MAX(v0.z, MAX(v1.z, v2.z));

	return AABB::createFromMinMax(min_, max_);
}

// http://tavianator.com/fast-branchless-raybounding-box-intersections-part-2-nans/
CUDA_CALLABLE bool AABB::intersects(const Ray& ray) const
{
	float tmin = 0;
	float tmax = 0;
	float tymin = 0;
	float tymax = 0;

	if (tmin > tymax || tymin > tmax)
		return false;

	if (tymin > tmin)
		tmin = tymin;

	if (tymax < tmax)
		tmax = tymax;

	float tzmin = 0;
	float tzmax = 0;

	if (tmin > tzmax || tzmin > tmax)
		return false;

	if (tzmin > tmin)
		tmin = tzmin;

	if (tzmax < tmax)
		tmax = tzmax;

	return (tmin < ray.maxDistance) && (tmax > ray.minDistance);
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

Vector3 AABB::getCenter() const
{
	return (min + max) * 0.5;
}

Vector3 AABB::getExtent() const
{
	return max - min;
}

float AABB::getSurfaceArea() const
{
	Vector3 extent = getExtent();
	return 2.0f * (extent.x * extent.y + extent.z * extent.y + extent.x * extent.z);
}
