// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Core/Sphere.h"
#include "Core/AABB.h"
#include "Core/Ray.h"
#include "Core/Intersection.h"

using namespace Varjo;

CUDA_CALLABLE bool Sphere::intersect(const Ray& ray, Intersection& intersection) const
{
	Vector3 rayOriginToSphere = position - ray.origin;
	float rayOriginToSphereDistance2 = rayOriginToSphere.lengthSquared();

	float t1 = rayOriginToSphere.dot(ray.direction);
	float sphereToRayDistance2 = rayOriginToSphereDistance2 - (t1 * t1);
	float radius2 = radius * radius;

	bool rayOriginIsOutside = (rayOriginToSphereDistance2 >= radius2);

	if (rayOriginIsOutside)
	{
		// whole sphere is behind the ray origin
		if (t1 < 0.0f)
			return false;

		// ray misses the sphere completely
		if (sphereToRayDistance2 > radius2)
			return false;
	}

	float t2 = sqrt(radius2 - sphereToRayDistance2);
	float t = (rayOriginIsOutside) ? (t1 - t2) : (t1 + t2);

	if (t < ray.minDistance || t > ray.maxDistance)
		return false;

	// there was another intersection closer to camera
	if (t > intersection.distance)
		return false;

	// intersection position and normal
	Vector3 ip = ray.origin + (t * ray.direction);
	Vector3 normal = (ip - position).normalized();

	intersection.wasFound = true;
	intersection.distance = t;
	intersection.position = ip;
	intersection.normal = normal;
	
	// spherical texture coordinate calculation
	intersection.texcoord.x = 0.5f + atan2(normal.z, normal.x) / (2.0f * float(M_PI));
	intersection.texcoord.y = 0.5f - asin(normal.y) / float(M_PI);

	return true;
}

AABB Sphere::getAABB() const
{
	return AABB::createFromCenterExtent(position, Vector3(radius * 2.0f, radius * 2.0f, radius * 2.0f));
}
