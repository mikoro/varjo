// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <cfloat>

#include "Math/Vector3.h"
#include "Math/Vector2.h"
#include "Core/Color.h"
#include "Math/ONB.h"

namespace Varjo
{
	class Intersection
	{
	public:

		bool wasFound = false;
		bool isBehind = false;
		bool hasColor = false;

		float distance = FLT_MAX;
		float area = 0.0f;

		Vector3 position;
		Vector3 normal;
		Vector2 texcoord;
		Color color;

		ONB onb;

		uint32_t materialIndex = 0;
	};
}
