// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include <GLFW/glfw3.h>

#include "tinyformat/tinyformat.h"

#include "Utils/FpsCounter.h"

using namespace Varjo;

FpsCounter::FpsCounter()
{
	lastTime = glfwGetTime();

	averageFrameTime.setAlpha(0.05f);
	averageFrameTime.setAverage(1.0f / 30.0f);
}

void FpsCounter::tick()
{
	double currentTime = glfwGetTime();
	frameTime = currentTime - lastTime;
	lastTime = currentTime;

	// prevent too large frametime changes
	if (frameTime > 2.0 * averageFrameTime.getAverage())
		frameTime = 2.0 * averageFrameTime.getAverage();
}

void FpsCounter::update()
{
	averageFrameTime.addMeasurement(float(frameTime));
}

float FpsCounter::getFrameTime() const
{
	return averageFrameTime.getAverage() * 1000.0f;
}

float FpsCounter::getFps() const
{
	return 1.0f / averageFrameTime.getAverage();
}

std::string FpsCounter::getFpsString() const
{
	return tfm::format("%.1f", getFps());
}
