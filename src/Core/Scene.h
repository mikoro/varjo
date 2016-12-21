// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <vector>

#include "Core/BVH.h"
#include "Core/Camera.h"

namespace Varjo
{
	class Scene
	{
	public:

		static Scene createTestScene1();

		void initialize();

		std::vector<Sphere> primitives;
		std::vector<BVHNode> nodes;
		Camera camera;
	};
}
