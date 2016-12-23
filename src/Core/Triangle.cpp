// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Core/Triangle.h"

using namespace Varjo;

AABB Triangle::getAABB() const
{
	return AABB::createFromVertices(vertices[0], vertices[1], vertices[2]);
}
