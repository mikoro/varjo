// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Core/Scene.h"

using namespace Varjo;

Scene Scene::createTestScene1()
{
	Scene scene;

	scene.camera.position = Vector3(0.01f, 0.01f, 10.0f);
	scene.camera.fov = 75.0f;

	scene.modelFileName = "data\\models\\cornellbox\\cornellbox.obj";

	return scene;
}

void Scene::initialize()
{
	camera.initialize();

	ModelLoader modelLoader;
	ModelLoaderOutput modelOutput = modelLoader.load(modelFileName);

	triangles = modelOutput.triangles;
	materials = modelOutput.materials;

	BVH::build(triangles, nodes);
	//BVH::exportDot(nodes, "bvh.gv");
}
