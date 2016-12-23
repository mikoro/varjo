// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <map>
#include <vector>

#include "Core/Triangle.h"
#include "Core/Material.h"

/*

Only OBJ (and MTL) files are supported.

Restrictions:
 - numbers cannot have scientific notation

*/

namespace Varjo
{
	struct ModelLoaderOutput
	{
		std::vector<Material> materials;
		std::vector<Triangle> triangles;
	};

	class ModelLoader
	{
	public:

		ModelLoaderOutput load(const std::string& fileName);

	private:

		void processMaterialFile(const std::string& rootDirectory, const std::string& mtlFileName, ModelLoaderOutput& output);
		bool processFace(const char* buffer, uint32_t lineStartIndex, uint32_t lineEndIndex, uint32_t lineNumber, ModelLoaderOutput& output);

		std::vector<float3> vertices;
		std::vector<float3> normals;
		std::vector<float2> texcoords;

		uint32_t currentMaterialIndex = 0;
		std::map<std::string, uint32_t> materialIndexMap;
	};
}
