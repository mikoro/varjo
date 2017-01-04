// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include <GL/gl3w.h>
#include <GLFW/glfw3.h>

#define NANOVG_GL3_IMPLEMENTATION
#include "nanovg/nanovg.h"
#include "nanovg/nanovg_gl.h"

#include "tinyformat/tinyformat.h"

#include "Utils/App.h"
#include "Utils/InfoPanel.h"
#include "Utils/Settings.h"
#include "Utils/StringUtils.h"
#include "Core/Scene.h"

using namespace Varjo;

InfoPanel::~InfoPanel()
{
	if (context != nullptr)
	{
		nvgDeleteGL3(context);
		context = nullptr;
	}
}

void InfoPanel::initialize()
{
	context = nvgCreateGL3(NVG_ANTIALIAS | NVG_STENCIL_STROKES);

	if (context == nullptr)
		throw std::runtime_error("Could not initialize NanoVG");

	nvgCreateFont(context, "mono", "data/fonts/RobotoMono-Regular.ttf");
}

void InfoPanel::render(const Scene& scene, const Renderer& renderer)
{
	if (currentState == InfoPanelState::OFF)
		return;

	GLFWwindow* window = App::getWindow().getGlfwWindow();
	int windowWidth, windowHeight, framebufferWidth, framebufferHeight;

	glfwGetWindowSize(window, &windowWidth, &windowHeight);
	glfwGetFramebufferSize(window, &framebufferWidth, &framebufferHeight);

	float pixelRatio = float(framebufferWidth) / float(windowWidth);

	nvgBeginFrame(context, windowWidth, windowHeight, pixelRatio);

	if (currentState == InfoPanelState::FPS)
		renderFps();
	else if (currentState == InfoPanelState::FULL)
		renderFull(scene, renderer);

	nvgEndFrame(context);
}

void InfoPanel::setState(InfoPanelState state)
{
	currentState = state;
}

void InfoPanel::selectNextState()
{
	if (currentState == InfoPanelState::OFF)
		currentState = InfoPanelState::FPS;
	else if (currentState == InfoPanelState::FPS)
		currentState = InfoPanelState::FULL;
	else if (currentState == InfoPanelState::FULL)
		currentState = InfoPanelState::OFF;
}

void InfoPanel::renderFps()
{
	Settings& settings = App::getSettings();
	const FpsCounter& fpsCounter = App::getWindow().getFpsCounter();

	std::string fpsString = tfm::format("%.1f", fpsCounter.getFps());

	nvgBeginPath(context);

	nvgFontSize(context, float(settings.general.infoPanelFontSize));
	nvgFontFace(context, "mono");

	float lineSpacing;
	nvgTextMetrics(context, nullptr, nullptr, &lineSpacing);
	
	float bounds[4];
	nvgTextBounds(context, 0.0f, 0.0f, "1234567890.", nullptr, bounds);
	float charWidth = (bounds[2] - bounds[0]) / 11.0f;
	float panelWidth = fpsString.length() * charWidth;
	float panelHeight = bounds[3] - bounds[1];
	float currentX = charWidth / 2.0f + 2.0f;
	float currentY = -bounds[1] + 2.0f;

	float rounding = 5.0f;
	float strokeWidth = 2.0f;

	nvgFillColor(context, nvgRGBA(0, 0, 0, 150));
	nvgRoundedRect(context, -rounding, -rounding, panelWidth + rounding + charWidth, panelHeight + rounding, rounding);
	nvgFill(context);

	nvgBeginPath(context);

	nvgStrokeWidth(context, strokeWidth);
	nvgStrokeColor(context, nvgRGBA(0, 0, 0, 180));
	nvgRoundedRect(context, -rounding, -rounding, panelWidth + rounding + charWidth + strokeWidth / 2.0f, panelHeight + rounding + strokeWidth / 2.0f, rounding + 1.0f);
	nvgStroke(context);

	nvgFillColor(context, nvgRGBA(255, 255, 255, 255));
	nvgText(context, currentX, currentY, fpsString.c_str(), nullptr);
}

void InfoPanel::renderFull(const Scene& scene, const Renderer& renderer)
{
	Settings& settings = App::getSettings();
	Window& window = App::getWindow();
	const Film& film = window.getFilm();
	const FpsCounter& fpsCounter = window.getFpsCounter();
	
	nvgBeginPath(context);

	nvgFontSize(context, float(settings.general.infoPanelFontSize));
	nvgFontFace(context, "mono");

	float lineSpacing;
	nvgTextMetrics(context, nullptr, nullptr, &lineSpacing);
	
	float bounds[4];
	nvgTextBounds(context, 0.0f, 0.0f, "1234567890.", nullptr, bounds);
	float charWidth = (bounds[2] - bounds[0]) / 11.0f;
	float panelWidth = 34 * charWidth;
	float panelHeight = 10 * lineSpacing + lineSpacing / 2.0f;
	float currentX = charWidth / 2.0f + 4.0f;
	float currentY = -bounds[1] + 4.0f;

	float rounding = 5.0f;
	float strokeWidth = 2.0f;

	nvgFillColor(context, nvgRGBA(0, 0, 0, 150));
	nvgRoundedRect(context, -rounding, -rounding, panelWidth + rounding + charWidth, panelHeight + rounding, rounding);
	nvgFill(context);

	nvgBeginPath(context);

	nvgStrokeWidth(context, strokeWidth);
	nvgStrokeColor(context, nvgRGBA(0, 0, 0, 180));
	nvgRoundedRect(context, -rounding, -rounding, panelWidth + rounding + charWidth + strokeWidth / 2.0f, panelHeight + rounding + strokeWidth / 2.0f, rounding + 1.0f);
	nvgStroke(context);

	nvgFillColor(context, nvgRGBA(255, 255, 255, 255));

	nvgText(context, currentX, currentY, tfm::format("FPS: %.1f", fpsCounter.getFps()).c_str(), nullptr);
	currentY += lineSpacing;

	nvgText(context, currentX, currentY, tfm::format("Frametime: %.1f ms", fpsCounter.getFrameTime()).c_str(), nullptr);
	currentY += lineSpacing;

	int tempWindowWidth, tempWindowHeight, tempFramebufferWidth, tempFramebufferHeight;
	GLFWwindow* glfwWindow = window.getGlfwWindow();

	glfwGetWindowSize(glfwWindow, &tempWindowWidth, &tempWindowHeight);
	glfwGetFramebufferSize(glfwWindow, &tempFramebufferWidth, &tempFramebufferHeight);

	nvgText(context, currentX, currentY, tfm::format("Window: %dx%d (%dx%d)", tempWindowWidth, tempWindowHeight, tempFramebufferWidth, tempFramebufferHeight).c_str(), nullptr);
	currentY += lineSpacing;

	float totalPixels = float(film.getWidth() * film.getHeight());

	nvgText(context, currentX, currentY, tfm::format("Film: %dx%d (%.2fx) (%s)", film.getWidth(), film.getHeight(), settings.general.filmScale, StringUtils::humanizeNumber(totalPixels)).c_str(), nullptr);
	currentY += lineSpacing;

	int32_t filmMouseX = std::max(int32_t(0), std::min(window.getMouseInfo().filmX, int32_t(film.getWidth() - 1)));
	int32_t filmMouseY = std::max(int32_t(0), std::min(window.getMouseInfo().filmY, int32_t(film.getHeight() - 1)));;
	int32_t filmMouseIndex = filmMouseY * film.getWidth() + filmMouseX;

	nvgText(context, currentX, currentY, tfm::format("Mouse: (%d, %d, %d)", filmMouseX, filmMouseY, filmMouseIndex).c_str(), nullptr);
	currentY += lineSpacing;

	nvgText(context, currentX, currentY, tfm::format("Position: (%.2f, %.2f, %.2f)", scene.camera.position.x, scene.camera.position.y, scene.camera.position.z).c_str(), nullptr);
	currentY += lineSpacing;

	Vector3 direction = scene.camera.orientation.getDirection();
	nvgText(context, currentX, currentY, tfm::format("Direction: (%.2f, %.2f, %.2f)", direction.x, direction.y, direction.z).c_str(), nullptr);
	currentY += lineSpacing;

	nvgText(context, currentX, currentY, tfm::format("Rotation: (%.2f, %.2f, %.2f)", scene.camera.orientation.pitch, scene.camera.orientation.yaw, scene.camera.orientation.roll).c_str(), nullptr);
	currentY += lineSpacing;

	nvgText(context, currentX, currentY, tfm::format("Paths/s: %s", StringUtils::humanizeNumber(renderer.getPathsPerSecond())).c_str(), nullptr);
	currentY += lineSpacing;

	nvgText(context, currentX, currentY, tfm::format("Rays/s: %s", StringUtils::humanizeNumber(renderer.getRaysPerSecond())).c_str(), nullptr);
	currentY += lineSpacing;
}
