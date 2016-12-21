// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <cuda_runtime.h>

namespace Varjo
{
	class Scene;
	struct Sphere;
	struct BVHNode;
	struct CameraData;

	class Renderer
	{
	public:

		void initialize(const Scene& scene);
		void shutdown();
		void update(const Scene& scene);
		void filmResized(uint32_t width, uint32_t height);
		void render();

	private:

		Sphere* primitives = nullptr;
		BVHNode* nodes = nullptr;
		CameraData* camera = nullptr;

		dim3 clearKernelBlockDim;
		dim3 clearKernelGridDim;

		dim3 traceKernelBlockDim;
		dim3 traceKernelGridDim;
	};
}
