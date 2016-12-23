// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <cuda_runtime.h>

#include "Core/Scene.h"
#include "Core/Camera.h"
#include "Core/BVH.h"
#include "Core/Triangle.h"
#include "Core/Material.h"
#include "Core/Path.h"
#include "Core/Pixel.h"

namespace Varjo
{
	class Renderer
	{
	public:

		void initialize(const Scene& scene);
		void shutdown();
		void update(const Scene& scene);
		void filmResized(uint32_t width, uint32_t height);
		void render();

	private:

		const uint32_t pathCount = 100000;

		CameraData* camera = nullptr;
		BVHNode* nodes = nullptr;
		Triangle* triangles = nullptr;
		Triangle* emitters = nullptr;
		Material* materials = nullptr;
		Path* paths = nullptr;
		Pixel* pixels = nullptr;
		
		dim3 traceKernelBlockSize;
		dim3 traceKernelGridSize;
	};
}
