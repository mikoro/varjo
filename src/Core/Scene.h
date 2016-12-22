// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <vector>

#include "Core/Camera.h"
#include "Core/Sphere.h"
#include "Core/BVH.h"

namespace Varjo
{
	class Scene
	{
	public:

		static Scene createTestScene1();

		void initialize();

		Camera camera;
		std::vector<Sphere> primitives;
		std::vector<BVHNode> nodes;
	};
}
