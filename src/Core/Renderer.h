// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

namespace Varjo
{
	class Scene;
	class Sphere;
	struct BVHNode;
	class Camera;

	class Renderer
	{
	public:

		void initialize(const Scene& scene);
		void shutdown();
		void update(const Scene& scene);
		void render();

	private:

		Sphere* primitives = nullptr;
		BVHNode* nodes = nullptr;
		Camera* camera = nullptr;
	};
}
