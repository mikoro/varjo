// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Core/Triangle.h"
#include "Cuda/Math.h"

using namespace Varjo;

void Triangle::initialize()
{
	float3 v0tov1 = vertices[1] - vertices[0];
	float3 v0tov2 = vertices[2] - vertices[0];
	float3 cp = cross(v0tov1, v0tov2);
	normal = normalize(cp);
	area = 0.5f * length(cp);
}

AABB Triangle::getAABB() const
{
	return AABB::createFromVertices(vertices[0], vertices[1], vertices[2]);
}
