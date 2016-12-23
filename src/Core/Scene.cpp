// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Core/Scene.h"

using namespace Varjo;

Scene Scene::createTestScene1()
{
	Scene scene;

	scene.camera.position = Vector3(1.4004f, 1.8210f, 2.8143f);
	scene.camera.orientation = EulerAngle(-18.3100f, 29.0997f, 0.0000f);
	scene.camera.fov = 75.00f;

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
