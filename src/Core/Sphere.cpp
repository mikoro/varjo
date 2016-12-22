// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Core/Sphere.h"

using namespace Varjo;

AABB Sphere::getAABB() const
{
	return AABB::createFromCenterExtent(position, Vector3(radius * 2.0f, radius * 2.0f, radius * 2.0f));
}
