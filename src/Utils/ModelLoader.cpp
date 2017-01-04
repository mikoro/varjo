// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include <boost/filesystem.hpp>

#include "Utils/ModelLoader.h"
#include "Utils/App.h"
#include "Utils/Log.h"
#include "Utils/Timer.h"
#include "Cuda/Math.h"

using namespace Varjo;
namespace bf = boost::filesystem;

namespace
{
	std::string getAbsolutePath(const std::string& rootDirectory, const std::string& relativeFileName)
	{
		bf::path tempPath(rootDirectory);
		tempPath.append(relativeFileName.begin(), relativeFileName.end());
		std::string tempPathString = tempPath.string();
		std::replace(tempPathString.begin(), tempPathString.end(), '\\', '/');
		return tempPathString;
	}

	bool getLine(const char* buffer, uint32_t bufferLength, uint32_t& lineStartIndex, uint32_t& lineEndIndex)
	{
		while (lineStartIndex < bufferLength)
		{
			char c = buffer[lineStartIndex];

			if (c != '\r' && c != '\n')
				break;

			lineStartIndex++;
		}

		if (lineStartIndex >= bufferLength)
			return false;

		lineEndIndex = lineStartIndex;

		while (lineEndIndex < bufferLength)
		{
			char c = buffer[lineEndIndex];

			if (c == '\r' || c == '\n')
				break;

			lineEndIndex++;
		}

		return true;
	}

	bool getWord(const char* buffer, uint32_t lineEndIndex, uint32_t& wordStartIndex, uint32_t& wordEndIndex)
	{
		while (wordStartIndex < lineEndIndex)
		{
			char c = buffer[wordStartIndex];

			if (c != ' ')
				break;

			wordStartIndex++;
		}

		if (wordStartIndex >= lineEndIndex)
			return false;

		wordEndIndex = wordStartIndex;

		while (wordEndIndex < lineEndIndex)
		{
			char c = buffer[wordEndIndex];

			if (c == ' ')
				break;

			wordEndIndex++;
		}

		return true;
	}

	int32_t getInt(const char* buffer, uint32_t& startIndex, uint32_t endIndex)
	{
		char c = 0;

		while (startIndex < endIndex)
		{
			c = buffer[startIndex];

			if (c != ' ' && c != '/')
				break;

			startIndex++;
		}

		if (startIndex >= endIndex)
			return 0;

		int32_t sign = 1;
		int32_t accumulator = 0;

		if (c == '-')
		{
			sign = -1;

			if (++startIndex >= endIndex)
				return 0;
		}

		c = buffer[startIndex];

		while (c >= '0' && c <= '9')
		{
			accumulator = accumulator * 10 + c - '0';

			if (++startIndex >= endIndex)
				return sign * accumulator;

			c = buffer[startIndex];
		}

		return sign * accumulator;
	}

	float getFloat(const char* buffer, uint32_t& startIndex, uint32_t endIndex)
	{
		char c = 0;

		while (startIndex < endIndex)
		{
			c = buffer[startIndex];

			if (c != ' ')
				break;

			startIndex++;
		}

		if (startIndex >= endIndex)
			return 0.0f;

		float sign = 1.0f;
		float accumulator = 0.0f;

		if (c == '-')
		{
			sign = -1.0f;

			if (++startIndex >= endIndex)
				return 0.0f;
		}

		c = buffer[startIndex];

		while (c >= '0' && c <= '9')
		{
			accumulator = accumulator * 10.0f + c - '0';

			if (++startIndex >= endIndex)
				return sign * accumulator;

			c = buffer[startIndex];
		}

		if (c == '.')
		{
			if (++startIndex >= endIndex)
				return sign * accumulator;

			float k = 0.1f;
			c = buffer[startIndex];

			while (c >= '0' && c <= '9')
			{
				accumulator += (c - '0') * k;
				k *= 0.1f;

				if (++startIndex >= endIndex)
					return sign * accumulator;

				c = buffer[startIndex];
			}
		}

		return sign * accumulator;
	}

	bool compareWord(const char* buffer, uint32_t wordStartIndex, uint32_t wordEndIndex, const char* otherWord)
	{
		uint32_t wordLength = wordEndIndex - wordStartIndex;
		uint32_t otherWordLength = uint32_t(strlen(otherWord));

		if (wordLength != otherWordLength)
			return false;

		for (uint32_t i = wordStartIndex; i < wordEndIndex; ++i)
		{
			if (buffer[i] != otherWord[i - wordStartIndex])
				return false;
		}

		return true;
	}

	void checkIndices(const char* buffer, uint32_t wordStartIndex, uint32_t wordEndIndex, bool& hasNormals, bool& hasTexcoords)
	{
		uint32_t slashCount = 0;
		uint32_t doubleSlashCount = 0;

		for (uint32_t i = wordStartIndex; i < wordEndIndex; ++i)
		{
			if (buffer[i] == '/')
			{
				slashCount++;

				if (i < wordEndIndex - 1)
				{
					if (buffer[i + 1] == '/')
						doubleSlashCount++;
				}
			}
		}

		hasNormals = (slashCount == 2);
		hasTexcoords = (slashCount > 0 && doubleSlashCount == 0);
	}

	void getIndices(const char* buffer, uint32_t wordStartIndex, uint32_t wordEndIndex, bool hasNormals, bool hasTexcoords, int32_t& vertexIndex, int32_t& normalIndex, int32_t& texcoordIndex)
	{
		vertexIndex = getInt(buffer, wordStartIndex, wordEndIndex);

		if (hasTexcoords)
			texcoordIndex = getInt(buffer, wordStartIndex, wordEndIndex);
		
		if (hasNormals)
			normalIndex = getInt(buffer, wordStartIndex, wordEndIndex);
	}
}

ModelLoaderOutput ModelLoader::load(const std::string& fileName)
{
	Log& log = App::getLog();

	log.logInfo("Reading OBJ file %s", fileName);

	Timer timer;
	std::string rootDirectory = bf::absolute(fileName).parent_path().string();
	ModelLoaderOutput output;

	vertices.clear();
	normals.clear();
	texcoords.clear();
	materialIndexMap.clear();

	currentMaterialIndex = 0;
	
	std::ifstream file(fileName, std::ios::in | std::ios::binary | std::ios::ate);

	if (!file.good())
		throw std::runtime_error(tfm::format("Could not open the OBJ file"));

	auto size = file.tellg();
	file.seekg(0, std::ios::beg);

	std::vector<char> fileBuffer(size);
	file.read(&fileBuffer[0], size);

	file.close();

	char* fileBufferPtr = &fileBuffer[0];
	uint32_t fileBufferLength = uint32_t(fileBuffer.size());
	uint32_t lineStartIndex = 0;
	uint32_t lineEndIndex = 0;
	uint32_t lineNumber = 0;

	while (getLine(fileBufferPtr, fileBufferLength, lineStartIndex, lineEndIndex))
	{
		lineNumber++;

		uint32_t wordStartIndex = lineStartIndex;
		uint32_t wordEndIndex = 0;

		getWord(fileBufferPtr, lineEndIndex, wordStartIndex, wordEndIndex);

		if (compareWord(fileBufferPtr, wordStartIndex, wordEndIndex, "f")) // face
		{
			if (!processFace(fileBufferPtr, lineStartIndex + 2, lineEndIndex, lineNumber, output))
				break;
		}
		else if (compareWord(fileBufferPtr, wordStartIndex, wordEndIndex, "v")) // vertex
		{
			float3 vertex;
			wordStartIndex += 2;

			vertex.x = getFloat(fileBufferPtr, wordStartIndex, lineEndIndex);
			vertex.y = getFloat(fileBufferPtr, wordStartIndex, lineEndIndex);
			vertex.z = getFloat(fileBufferPtr, wordStartIndex, lineEndIndex);

			vertices.push_back(vertex);
		}
		else if (compareWord(fileBufferPtr, wordStartIndex, wordEndIndex, "vn")) // normal
		{
			float3 normal;
			wordStartIndex += 3;

			normal.x = getFloat(fileBufferPtr, wordStartIndex, lineEndIndex);
			normal.y = getFloat(fileBufferPtr, wordStartIndex, lineEndIndex);
			normal.z = getFloat(fileBufferPtr, wordStartIndex, lineEndIndex);

			normals.push_back(normal);
		}
		else if (compareWord(fileBufferPtr, wordStartIndex, wordEndIndex, "vt")) // texcoord
		{
			float2 texcoord;
			wordStartIndex += 3;

			texcoord.x = getFloat(fileBufferPtr, wordStartIndex, lineEndIndex);
			texcoord.y = getFloat(fileBufferPtr, wordStartIndex, lineEndIndex);

			texcoords.push_back(texcoord);
		}
		else if (compareWord(fileBufferPtr, wordStartIndex, wordEndIndex, "usemtl")) // select material
		{
			wordStartIndex = wordEndIndex;
			getWord(fileBufferPtr, lineEndIndex, wordStartIndex, wordEndIndex);
			std::string materialName(fileBufferPtr + wordStartIndex, wordEndIndex - wordStartIndex);

			if (materialIndexMap.count(materialName))
				currentMaterialIndex = materialIndexMap[materialName];
			else
			{
				log.logWarning("Could not find material named \"%s\"", materialName);
				currentMaterialIndex = 0;
			}
		}
		else if (compareWord(fileBufferPtr, wordStartIndex, wordEndIndex, "mtllib")) // new material file
		{
			wordStartIndex = wordEndIndex;
			getWord(fileBufferPtr, lineEndIndex, wordStartIndex, wordEndIndex);
			std::string mtlFileName(fileBufferPtr + wordStartIndex, wordEndIndex - wordStartIndex);

			processMaterialFile(rootDirectory, mtlFileName, output);
		}
		
		lineStartIndex = lineEndIndex;
	}

	log.logInfo("OBJ file reading finished (time: %s, vertices: %s, normals: %s, texcoords: %s, triangles: %s, materials: %s)", timer.getElapsed().getString(true), vertices.size(), normals.size(), texcoords.size(), output.triangles.size(), output.materials.size());

	return output;
}

void ModelLoader::processMaterialFile(const std::string& rootDirectory, const std::string& mtlFileName, ModelLoaderOutput& output)
{
	std::string absoluteMtlFileName = getAbsolutePath(rootDirectory, mtlFileName);
	App::getLog().logInfo("Reading MTL file (%s)", absoluteMtlFileName);
	std::ifstream file(absoluteMtlFileName);

	if (!file.good())
		throw std::runtime_error("Could not open the MTL file");

	Material currentMaterial;
	bool materialPending = false;

	std::string line;
	std::string part;
	std::string currentMaterialName;

	while (std::getline(file, line))
	{
		std::stringstream ss(line);
		ss >> part;

		if (part == "newmtl") // new material
		{
			if (materialPending)
				output.materials.push_back(currentMaterial);

			materialPending = true;
			currentMaterial = Material();
			ss >> currentMaterialName;
			materialIndexMap[currentMaterialName] = uint32_t(output.materials.size());
		}
		else if (part == "baseColor" || part == "Kd")
		{
			ss >> currentMaterial.baseColor.x;
			ss >> currentMaterial.baseColor.y;
			ss >> currentMaterial.baseColor.z;
		}
		else if (part == "emittance" || part == "Ke")
		{
			ss >> currentMaterial.emittance.x;
			ss >> currentMaterial.emittance.y;
			ss >> currentMaterial.emittance.z;
		}
		else if (part == "subsurface")
			ss >> currentMaterial.subsurface;
		else if (part == "metallic")
			ss >> currentMaterial.metallic;
		else if (part == "specular")
			ss >> currentMaterial.specular;
		else if (part == "specularTint")
			ss >> currentMaterial.specularTint;
		else if (part == "roughness")
			ss >> currentMaterial.roughness;
		else if (part == "anisotropic")
			ss >> currentMaterial.anisotropic;
		else if (part == "sheen")
			ss >> currentMaterial.sheen;
		else if (part == "sheenTint")
			ss >> currentMaterial.sheenTint;
		else if (part == "clearcoat")
			ss >> currentMaterial.clearcoat;
		else if (part == "clearcoatGloss")
			ss >> currentMaterial.clearcoatGloss;
	}

	file.close();

	if (materialPending)
		output.materials.push_back(currentMaterial);
}

bool ModelLoader::processFace(const char* buffer, uint32_t lineStartIndex, uint32_t lineEndIndex, uint32_t lineNumber, ModelLoaderOutput& output)
{
	Log& log = App::getLog();

	uint32_t vertexIndices[4];
	uint32_t normalIndices[4];
	uint32_t texcoordIndices[4];
	uint32_t vertexCount = 0;

	bool hasNormals = false;
	bool hasTexcoords = false;

	uint32_t wordStartIndex = lineStartIndex;
	uint32_t wordEndIndex = 0;

	for (uint32_t i = 0; i < 4; ++i)
	{
		if (!getWord(buffer, lineEndIndex, wordStartIndex, wordEndIndex))
			break;

		if (i == 0)
			checkIndices(buffer, wordStartIndex, wordEndIndex, hasNormals, hasTexcoords);

		vertexCount++;

		int32_t vertexIndex;
		int32_t texcoordIndex;
		int32_t normalIndex;

		getIndices(buffer, wordStartIndex, wordEndIndex, hasNormals, hasTexcoords, vertexIndex, normalIndex, texcoordIndex);

		if (vertexIndex < 0)
			vertexIndex = int32_t(vertices.size()) + vertexIndex;
		else
			vertexIndex--;

		if (vertexIndex < 0 || vertexIndex >= int32_t(vertices.size()))
		{
			log.logWarning("Vertex index (%s) was out of bounds (line: %s)", vertexIndex, lineNumber);
			return false;
		}

		vertexIndices[i] = uint32_t(vertexIndex);

		if (hasTexcoords)
		{
			if (texcoordIndex < 0)
				texcoordIndex = int32_t(texcoords.size()) + texcoordIndex;
			else
				texcoordIndex--;

			if (texcoordIndex < 0 || texcoordIndex >= int32_t(texcoords.size()))
			{
				log.logWarning("Texcoord index (%s) was out of bounds (line: %s)", texcoordIndex, lineNumber);
				return false;
			}

			texcoordIndices[i] = uint32_t(texcoordIndex);
		}

		if (hasNormals)
		{
			if (normalIndex < 0)
				normalIndex = int32_t(normals.size()) + normalIndex;
			else
				normalIndex--;

			if (normalIndex < 0 || normalIndex >= int32_t(normals.size()))
			{
				log.logWarning("Normal index (%s) was out of bounds (line: %s)", normalIndex, lineNumber);
				return false;
			}

			normalIndices[i] = uint32_t(normalIndex);
		}

		wordStartIndex = wordEndIndex;
	}

	if (vertexCount < 3)
	{
		log.logWarning("Too few vertices (%s) in a face (line: %s)", vertexCount, lineNumber);
		return false;
	}

	// triangulate
	for (uint32_t i = 2; i < vertexCount; ++i)
	{
		Triangle triangle;
		triangle.materialIndex = currentMaterialIndex;

		triangle.vertices[0] = vertices[vertexIndices[0]];
		triangle.vertices[1] = vertices[vertexIndices[i - 1]];
		triangle.vertices[2] = vertices[vertexIndices[i]];

		if (hasNormals)
		{
			triangle.normals[0] = normals[normalIndices[0]];
			triangle.normals[1] = normals[normalIndices[i - 1]];
			triangle.normals[2] = normals[normalIndices[i]];
		}
		else
		{
			float3 v0tov1 = triangle.vertices[1] - triangle.vertices[0];
			float3 v0tov2 = triangle.vertices[2] - triangle.vertices[0];
			float3 normal = normalize(cross(v0tov1, v0tov2));

			triangle.normals[0] = triangle.normals[1] = triangle.normals[2] = normal;
		}

		if (hasTexcoords)
		{
			triangle.texcoords[0] = texcoords[texcoordIndices[0]];
			triangle.texcoords[1] = texcoords[texcoordIndices[i - 1]];
			triangle.texcoords[2] = texcoords[texcoordIndices[i]];
		}

		output.triangles.push_back(triangle);
	}

	return true;
}
