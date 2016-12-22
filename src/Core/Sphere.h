// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "Core/AABB.h"
#include "Math/Vector3.h"

namespace Varjo
{
	class Sphere
	{
	public:

		AABB getAABB() const;

		Vector3 position;
		float radius;
	};
}
