// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <cfloat>

#include "Math/Vector3.h"
#include "Math/Vector2.h"

namespace Varjo
{
	class Intersection
	{
	public:

		float distance = FLT_MAX;
		
		Vector3 position;
		Vector3 normal;
		Vector2 texcoord;
	};
}
