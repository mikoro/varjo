// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Core/Sphere.h"
#include "Core/AABB.h"
#include "Core/Ray.h"
#include "Core/Intersection.h"

using namespace Varjo;

AABB Sphere::getAABB() const
{
	return AABB::createFromCenterExtent(Vector3(position.x, position.y, position.z), Vector3(radius * 2.0f, radius * 2.0f, radius * 2.0f));
}
