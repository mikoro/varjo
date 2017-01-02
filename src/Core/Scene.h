// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <vector>

#include "Core/Camera.h"
#include "Core/BVH.h"
#include "Core/Triangle.h"
#include "Core/Material.h"
#include "Utils/ModelLoader.h"

namespace Varjo
{
	class Scene
	{
	public:

		static Scene createTestScene1();

		void initialize();

		Camera camera;
		std::vector<BVHNode> nodes;
		std::vector<Triangle> triangles;
		std::vector<uint32_t> emitters;
		std::vector<Material> materials;
		std::string modelFileName;
	};
}
