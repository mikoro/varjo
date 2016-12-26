// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

struct NVGcontext;

namespace Varjo
{
	class Scene;
	
	enum class InfoPanelState { OFF, FPS, FULL };

	class InfoPanel
	{
	public:

		~InfoPanel();

		void initialize();
		void render(const Scene& scene, const Renderer& renderer);

		void setState(InfoPanelState state);
		void selectNextState();

	private:

		void renderFps();
		void renderFull(const Scene& scene, const Renderer& renderer);

		NVGcontext* context = nullptr;
		InfoPanelState currentState = InfoPanelState::OFF;
	};
}
