// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include <GL/gl3w.h>
#include <GLFW/glfw3.h>

#include "Utils/Window.h"
#include "Utils/GLUtils.h"
#include "Utils/Log.h"
#include "Utils/Settings.h"
#include "Utils/CudaUtils.h"
#include "Utils/App.h"

using namespace Varjo;

namespace
{
	void glfwErrorCallback(int32_t error, const char* description)
	{
		App::getLog().logError("GLFW error (%s): %s", error, description);
	}

	MouseInfo* mouseInfoPtr = nullptr;

	void glfwMouseWheelScroll(GLFWwindow* window, double xoffset, double yoffset)
	{
		(void)window;
		(void)xoffset;

		mouseInfoPtr->scrollY = float(yoffset);
		mouseInfoPtr->hasScrolled = true;
	}
}

Window::~Window()
{
	if (glfwInitialized)
		glfwTerminate();
}

void Window::run()
{
	Log& log = App::getLog();
	Settings& settings = App::getSettings();

	log.logInfo("Initializing GLFW library");

	glfwSetErrorCallback(::glfwErrorCallback);

	if (!glfwInit())
		throw std::runtime_error("Could not initialize GLFW library");

	glfwInitialized = true;
	startTime = glfwGetTime();

	log.logInfo("Creating window and OpenGL context (%sx%s, fullscreen: %s)", settings.window.width, settings.window.height, settings.window.fullscreen);

	glfwWindow = glfwCreateWindow(int32_t(settings.window.width), int32_t(settings.window.height), "Varjo", settings.window.fullscreen ? glfwGetPrimaryMonitor() : nullptr, nullptr);

	if (!glfwWindow)
		throw std::runtime_error("Could not create the window");

	printWindowSize();
	glfwSetScrollCallback(glfwWindow, ::glfwMouseWheelScroll);
	mouseInfoPtr = &mouseInfo;

	const GLFWvidmode* videoMode = glfwGetVideoMode(glfwGetPrimaryMonitor());
	glfwSetWindowPos(glfwWindow, (videoMode->width / 2 - int32_t(settings.window.width) / 2), (videoMode->height / 2 - int32_t(settings.window.height) / 2));
	glfwMakeContextCurrent(glfwWindow);

	log.logInfo("Initializing GL3W library");

	int32_t result = gl3wInit();

	if (result == -1)
		throw std::runtime_error("Could not initialize GL3W library");

	log.logInfo("OpenGL Vendor: %s | Renderer: %s | Version: %s | GLSL: %s", glGetString(GL_VENDOR), glGetString(GL_RENDERER), glGetString(GL_VERSION), glGetString(GL_SHADING_LANGUAGE_VERSION));

	glfwSwapInterval(settings.window.vsync ? 1 : 0);

	if (settings.window.hideCursor)
		glfwSetInputMode(glfwWindow, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	windowWidth = settings.window.width;
	windowHeight = settings.window.height;

	glViewport(0, 0, GLsizei(windowWidth), GLsizei(windowHeight));

	mainloop();
}

uint32_t Window::getWindowWidth() const
{
	return windowWidth;
}

uint32_t Window::getWindowHeight() const
{
	return windowHeight;
}

const MouseInfo& Window::getMouseInfo() const
{
	return mouseInfo;
}

float Window::getElapsedTime() const
{
	return float(glfwGetTime() - startTime);
}

const FpsCounter& Window::getFpsCounter() const
{
	return fpsCounter;
}

bool Window::keyIsDown(int32_t key)
{
	return (glfwGetKey(glfwWindow, key) == GLFW_PRESS);
}

bool Window::mouseIsDown(int32_t button)
{
	return (glfwGetMouseButton(glfwWindow, button) == GLFW_PRESS);
}

bool Window::keyWasPressed(int32_t key)
{
	if (keyIsDown(key))
	{
		if (!keyStates[key])
		{
			keyStates[key] = true;
			return true;
		}
	}
	else
		keyStates[key] = false;

	return false;
}

bool Window::mouseWasPressed(int32_t button)
{
	if (mouseIsDown(button))
	{
		if (!mouseStates[button])
		{
			mouseStates[button] = true;
			return true;
		}
	}
	else
		mouseStates[button] = false;

	return false;
}

float Window::getMouseWheelScroll()
{
	if (mouseInfo.hasScrolled)
	{
		mouseInfo.hasScrolled = false;
		return mouseInfo.scrollY;
	}

	return 0.0f;
}

// http://gafferongames.com/game-physics/fix-your-timestep/
// http://gamesfromwithin.com/casey-and-the-clearly-deterministic-contraptions
// https://randomascii.wordpress.com/2012/02/13/dont-store-that-in-a-float/
void Window::mainloop()
{
	App::getLog().logInfo("Entering the main loop");

	double timeStep = 1.0 / 60.0;
	double previousTime = glfwGetTime();
	double timeAccumulator = 0.0;

	update(0.0f);

	while (shouldRun)
	{
		double currentTime = glfwGetTime();
		double frameTime = currentTime - previousTime;
		previousTime = currentTime;

		// prevent too large frametimes (e.g. program was paused)
		if (frameTime > 0.25)
			frameTime = 0.25;

		timeAccumulator += frameTime;

		while (timeAccumulator >= timeStep)
		{
			update(float(timeStep));
			timeAccumulator -= timeStep;
		}

		double interpolation = timeAccumulator / timeStep;
		render(float(frameTime), float(interpolation));
	}
}

void Window::update(float timeStep)
{
	Settings& settings = App::getSettings();
	
	fpsCounter.update();
	glfwPollEvents();

	checkWindowSize();

	double newMouseX, newMouseY;
	glfwGetCursorPos(glfwWindow, &newMouseX, &newMouseY);

	mouseInfo.windowX = int32_t(newMouseX + 0.5);
	mouseInfo.windowY = int32_t(double(windowHeight) - newMouseY - 1.0 + 0.5);
	mouseInfo.filmX = int32_t((mouseInfo.windowX / double(windowWidth)) * (double(windowWidth) * settings.window.renderScale) + 0.5);
	mouseInfo.filmY = int32_t((mouseInfo.windowY / double(windowHeight)) * (double(windowHeight) * settings.window.renderScale) + 0.5);
	mouseInfo.deltaX = mouseInfo.windowX - previousMouseX;
	mouseInfo.deltaY = mouseInfo.windowY - previousMouseY;
	previousMouseX = mouseInfo.windowX;
	previousMouseY = mouseInfo.windowY;

	if (glfwWindowShouldClose(glfwWindow))
		shouldRun = false;

	if (keyWasPressed(GLFW_KEY_ESCAPE))
		shouldRun = false;

	// update
	(void)timeStep;
}

void Window::render(float timeStep, float interpolation)
{
	fpsCounter.tick();

	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

	// render
	(void)timeStep;
	(void)interpolation;

	glfwSwapBuffers(glfwWindow);
}

void Window::checkWindowSize()
{
	int tempWindowWidth, tempWindowHeight;

	glfwGetFramebufferSize(glfwWindow, &tempWindowWidth, &tempWindowHeight);

	if (tempWindowWidth == 0 || tempWindowHeight == 0)
		return;

	if (uint32_t(tempWindowWidth) != windowWidth || uint32_t(tempWindowHeight) != windowHeight)
	{
		printWindowSize();
		windowResized(uint32_t(tempWindowWidth), uint32_t(tempWindowHeight));
	}
}

void Window::printWindowSize()
{
	int tempWindowWidth, tempWindowHeight, tempFramebufferWidth, tempFramebufferHeight;

	glfwGetWindowSize(glfwWindow, &tempWindowWidth, &tempWindowHeight);
	glfwGetFramebufferSize(glfwWindow, &tempFramebufferWidth, &tempFramebufferHeight);

	App::getLog().logInfo("GLFW window size: %dx%d | GLFW framebuffer size: %dx%d", tempWindowWidth, tempWindowHeight, tempFramebufferWidth, tempFramebufferHeight);
}

void Window::windowResized(uint32_t width, uint32_t height)
{
	windowWidth = width;
	windowHeight = height;

	glViewport(0, 0, GLsizei(windowWidth), GLsizei(windowHeight));
}
