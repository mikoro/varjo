// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "Math/MovingAverage.h"

namespace Varjo
{
	class FpsCounter
	{
	public:

		FpsCounter();

		void tick();
		void update();
		float getFrameTime() const;
		float getFps() const;
		std::string getFpsString() const;

	private:

		double lastTime = 0.0;
		double frameTime = 0.0;
		MovingAverage averageFrameTime;
	};
}
