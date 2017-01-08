// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "Core/Scene.h"
#include "Core/Camera.h"
#include "Core/BVH.h"
#include "Core/Triangle.h"
#include "Core/Material.h"
#include "Cuda/Structs.h"
#include "Utils/Timer.h"
#include "Math/MovingAverage.h"

#define RAY_EPSILON 0.0001f

namespace Varjo
{
	class Renderer
	{
	public:

		void initialize(const Scene& scene);
		void shutdown();
		void update(const Scene& scene);
		void filmResized(uint32_t filmWidth, uint32_t filmHeight);
		void clear();
		void render();
		float getPathsPerSecond() const;
		float getRaysPerSecond() const;

	private:

		const uint32_t pathCount = 1000000;

		CameraData* camera = nullptr;
		BVHNode* nodes = nullptr;
		Triangle* triangles = nullptr;
		uint32_t* emitters = nullptr;
		Material* materials = nullptr;
		Paths* paths = nullptr;
		Queues* queues = nullptr;
		Pixel* pixels = nullptr;
		
		uint32_t emitterCount = 0;
		uint32_t pixelCount = 0;

		int initPathsBlockSize = 0;
		int initPathsGridSize = 0;
		int clearPathsBlockSize = 0;
		int clearPathsGridSize = 0;
		int clearPixelsBlockSize = 0;
		int clearPixelsGridSize = 0;
		int logicBlockSize = 0;
		int logicGridSize = 0;
		int writePixelsBlockSize = 0;
		int writePixelsGridSize = 0;
		int newPathBlockSize = 0;
		int newPathGridSize = 0;
		int diffuseMaterialBlockSize = 0;
		int diffuseMaterialGridSize = 0;
		int disneyMaterialBlockSize = 0;
		int disneyMaterialGridSize = 0;
		int extensionRayBlockSize = 0;
		int extensionRayGridSize = 0;
		int lightRayBlockSize = 0;
		int lightRayGridSize = 0;
		int writeFilmBlockSize = 0;
		int writeFilmGridSize = 0;

		Timer timer;
		MovingAverage averagePathsPerSecond;
		MovingAverage averageRaysPerSecond;
	};
}
