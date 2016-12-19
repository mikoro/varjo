// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Core/Scene.h"

using namespace Varjo;

Scene Scene::createTestScene1()
{
	Scene scene;

	for (int y = -10; y <= 10; y += 2)
	{
		for (int x = -10; x <= 10; x += 2)
		{
			Sphere sphere;
			sphere.position = Vector3(float(x), float(y), 0.0f);
			sphere.radius = 1.0f;
			scene.primitives.push_back(sphere);
		}
	}

	scene.camera.position = Vector3(0.0f, 0.0f, 10.0f);
	scene.camera.fov = 75.0f;

	return scene;
}

void Scene::initialize()
{
	BVH::build(nodes, primitives);

	camera.initialize();
}
