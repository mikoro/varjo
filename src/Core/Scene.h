// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <vector>

#include "Core/Camera.h"
#include "Core/Sphere.h"

namespace Varjo
{
	class Scene
	{
	public:

		Camera camera;

		std::vector<Sphere> spheres;
	};
}
