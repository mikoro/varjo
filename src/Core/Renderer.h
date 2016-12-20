// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <device_launch_parameters.h>

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

		dim3 clearKernelBlockDim;
		dim3 clearKernelGridDim;

		dim3 traceKernelBlockDim;
		dim3 traceKernelGridDim;
	};
}
