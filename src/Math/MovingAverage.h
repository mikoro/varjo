// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "Common.h"

namespace Varjo
{
	class MovingAverage
	{
	public:

		explicit MovingAverage(float alpha = 1.0f, float average = 0.0f);

		void setAlpha(float alpha);
		void setAverage(float average);
		void addMeasurement(float value);
		float getAverage() const;

	private:

		float alpha;
		float average;
	};
}
