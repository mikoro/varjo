// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Core/Scene.h"

using namespace Varjo;

Scene Scene::createTestScene1()
{
	Scene scene;

	scene.camera.position = Vector3(0.0f, 0.0f, 100.0f);
	scene.camera.fov = 75.0f;

	//scene.camera.position = Vector3(1000.0f, 1000.0f, 0.0f);
	//scene.camera.orientation = EulerAngle(-45.5f, 90.0f, 0.0f);
	//scene.camera.fov = 75.00f;

	for (int y = -100; y <= 100; y += 2)
	{
		for (int x = -100; x <= 100; x += 2)
		{
			Sphere sphere;
			sphere.position = Vector3(float(x), float(y), 0.0f);
			sphere.radius = 1.0f;
			scene.primitives.push_back(sphere);
		}
	}

	return scene;
}

void Scene::initialize()
{
	camera.initialize();
	BVH::build(primitives, nodes);
	BVH::exportDot(nodes, "bvh.gv");
}
