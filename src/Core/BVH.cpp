// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Core/BVH.h"
#include "Utils/Log.h"
#include "Utils/Timer.h"
#include "Core/Scene.h"
#include "Utils/App.h"

#ifdef _WIN32
#include <ppl.h>
#define PARALLEL_SORT concurrency::parallel_sort
#endif

#ifdef __linux
#include <parallel/algorithm>
#define PARALLEL_SORT __gnu_parallel::sort
#endif

using namespace Varjo;

namespace
{
	struct BVHBuildTriangle
	{
		Triangle* triangle;
		AABB aabb;
		float3 center;
	};

	struct BVHSplitCache
	{
		AABB aabb;
		float cost;
	};

	struct BVHBuildEntry
	{
		uint32_t start;
		uint32_t end;
		int32_t parent;
	};

	struct BVHSplitOutput
	{
		uint32_t index = 0;
		uint32_t axis = 0;
		AABB fullAABB;
		AABB leftAABB;
		AABB rightAABB;
	};

	BVHSplitOutput calculateSplit(std::vector<BVHBuildTriangle>& buildTriangles, std::vector<BVHSplitCache>& splitCache, uint32_t start, uint32_t end)
	{
		assert(end > start);

		BVHSplitOutput output;
		float lowestCost = FLT_MAX;
		AABB fullAABB[3];

		for (uint32_t axis = 0; axis <= 2; ++axis)
		{
			PARALLEL_SORT(buildTriangles.begin() + start, buildTriangles.begin() + end, [axis](const BVHBuildTriangle& t1, const BVHBuildTriangle& t2)
			{
				return (&t1.center.x)[axis] < (&t2.center.x)[axis];
			});

			AABB rightAABB;
			uint32_t rightCount = 0;

			for (int32_t i = end - 1; i >= int32_t(start); --i)
			{
				rightAABB.expand(buildTriangles[i].aabb);
				rightCount++;

				splitCache[i].aabb = rightAABB;
				splitCache[i].cost = rightAABB.getSurfaceArea() * float(rightCount);
			}

			AABB leftAABB;
			uint32_t leftCount = 0;

			for (uint32_t i = start; i < end; ++i)
			{
				leftAABB.expand(buildTriangles[i].aabb);
				leftCount++;

				float cost = leftAABB.getSurfaceArea() * float(leftCount);

				if (i + 1 < end)
					cost += splitCache[i + 1].cost;

				if (cost < lowestCost)
				{
					output.index = i + 1;
					output.axis = axis;
					output.leftAABB = leftAABB;

					if (output.index < end)
						output.rightAABB = splitCache[output.index].aabb;

					lowestCost = cost;
				}
			}

			fullAABB[axis] = leftAABB;
		}

		assert(output.index >= start && output.index <= end);

		if (output.axis != 2)
		{
			PARALLEL_SORT(buildTriangles.begin() + start, buildTriangles.begin() + end, [output](const BVHBuildTriangle& t1, const BVHBuildTriangle& t2)
			{
				return (&t1.center.x)[output.axis] < (&t2.center.x)[output.axis];
			});
		}

		output.fullAABB = fullAABB[output.axis];
		return output;
	}

}

void BVH::build(std::vector<Triangle>& triangles, std::vector<BVHNode>& nodes)
{
	Log& log = App::getLog();

	Timer timer;
	uint32_t triangleCount = uint32_t(triangles.size());

	if (triangleCount == 0)
	{
		log.logWarning("Could not build BVH from empty triangles list");
		return;
	}

	log.logInfo("BVH building started (triangles: %d)", triangleCount);

	std::vector<BVHBuildTriangle> buildTriangles(triangleCount);
	std::vector<BVHSplitCache> splitCache(triangleCount);
	std::vector<uint32_t> leafTriangleCounts;
	BVHSplitOutput splitOutput;

	for (uint32_t i = 0; i < triangleCount; ++i)
	{
		AABB aabb = triangles[i].getAABB();

		buildTriangles[i].triangle = &triangles[i];
		buildTriangles[i].aabb = aabb;
		buildTriangles[i].center = aabb.getCenter();
	}

	nodes.clear();
	nodes.reserve(triangleCount);
	leafTriangleCounts.reserve(triangleCount / 4);

	BVHBuildEntry stack[128];
	uint32_t stackIndex = 0;
	uint32_t nodeCount = 0;

	// push to stack
	stack[stackIndex].start = 0;
	stack[stackIndex].end = triangleCount;
	stack[stackIndex].parent = -1;
	stackIndex++;

	while (stackIndex > 0)
	{
		nodeCount++;

		// pop from stack
		BVHBuildEntry buildEntry = stack[--stackIndex];

		BVHNode node;
		node.rightOffset = -3;
		node.triangleOffset = uint32_t(buildEntry.start);
		node.triangleCount = uint32_t(buildEntry.end - buildEntry.start);

		// set as leaf node
		if (node.triangleCount <= 4)
			node.rightOffset = 0;

		// update the parent rightOffset when visiting its right child
		if (buildEntry.parent != -1) // not root
		{
			uint32_t parent = uint32_t(buildEntry.parent);

			if (++nodes[parent].rightOffset == -1)
				nodes[parent].rightOffset = int32_t(nodeCount - 1 - parent);
		}

		// not a leaf node -> split
		if (node.rightOffset != 0)
		{
			splitOutput = calculateSplit(buildTriangles, splitCache, buildEntry.start, buildEntry.end);
			node.aabb = splitOutput.fullAABB;
		}

		nodes.push_back(node);

		if (node.rightOffset == 0)
		{
			leafTriangleCounts.push_back(node.triangleCount);
			continue;
		}

		// push right child
		stack[stackIndex].start = splitOutput.index;
		stack[stackIndex].end = buildEntry.end;
		stack[stackIndex].parent = int32_t(nodeCount) - 1;
		stackIndex++;

		// push left child
		stack[stackIndex].start = buildEntry.start;
		stack[stackIndex].end = splitOutput.index;
		stack[stackIndex].parent = int32_t(nodeCount) - 1;
		stackIndex++;
	}

	std::vector<Triangle> sortedTriangles(triangleCount);

	for (uint32_t i = 0; i < triangleCount; ++i)
		sortedTriangles[i] = *buildTriangles[i].triangle;

	triangles = sortedTriangles;

	float mean = float(triangleCount) / float(leafTriangleCounts.size());
	float std = 0.0f;

	for (uint32_t i = 0; i < leafTriangleCounts.size(); ++i)
		std += std::pow(float(leafTriangleCounts[i]) - mean, 2);

	std /= float(leafTriangleCounts.size());

	log.logInfo("BVH building finished (time: %s, nodes: %d, leafs: %d, triangles/leaf: %.2f +- %.2f)", timer.getElapsed().getString(true), nodeCount, leafTriangleCounts.size(), mean, std);
}

void BVH::exportDot(std::vector<BVHNode>& nodes, const std::string& fileName)
{
	App::getLog().logInfo("Exporting BVH structure to %s", fileName);

	std::ofstream file(fileName);

	if (!file.is_open())
		throw std::runtime_error("Could not open file");

	uint32_t stack[16];
	uint32_t stackIndex = 1;
	stack[0] = 0;

	file << "digraph G {" << std::endl;
	
	while (stackIndex > 0)
	{
		uint32_t nodeIndex = stack[--stackIndex];
		const BVHNode& node = nodes[nodeIndex];

		// leaf node
		if (node.rightOffset == 0)
		{
			file << tfm::format("n%d -> { ", nodeIndex);

			for (uint32_t i = 0; i < node.triangleCount; ++i)
				file << tfm::format("t%d; ", node.triangleOffset + i);

			file << "}" << std::endl;

			continue;
		}

		uint32_t leftChild = nodeIndex + 1;
		uint32_t rightChild = nodeIndex + uint32_t(node.rightOffset);

		file << tfm::format("n%d -> { n%d; n%d }", nodeIndex, leftChild, rightChild) << std::endl;;

		stack[stackIndex++] = leftChild;
		stack[stackIndex++] = rightChild;
	}

	file << "}" << std::endl;
}
