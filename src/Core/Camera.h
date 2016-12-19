// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "Common.h"
#include "Core/Ray.h"
#include "Math/Vector2.h"
#include "Math/Vector3.h"
#include "Math/EulerAngle.h"

namespace Varjo
{
	class Random;
	class ONB;

	class Camera
	{
	public:

		void initialize();
		void setFilmSize(uint32_t width, uint32_t height);
		void update(float timeStep);
		void reset();
		bool isMoving() const;
		void saveState(const std::string& fileName) const;

		CUDA_CALLABLE Ray getRay(const Vector2& pointOnFilm) const;
		Vector3 getRight() const;
		Vector3 getUp() const;
		Vector3 getForward() const;

		Vector3 position;
		EulerAngle orientation;

		float fov = 75.0f;
		float moveSpeed = 10.0f;
		float mouseSpeed = 40.0f;
		float moveDrag = 3.0f;
		float mouseDrag = 6.0f;
		float autoStopSpeed = 0.01f;
		float slowSpeedModifier = 0.25f;
		float fastSpeedModifier = 2.5f;
		float veryFastSpeedModifier = 5.0f;

		bool enableMovement = true;
		bool smoothMovement = true;
		bool freeLook = false;

	private:

		bool cameraIsMoving = false;
		float currentSpeedModifier = 1.0f;

		Vector3 right;
		Vector3 up;
		Vector3 forward;
		Vector3 filmCenter;
		float filmWidth = 0.0f;
		float filmHeight = 0.0f;
		
		Vector3 velocity;
		Vector3 smoothVelocity;
		Vector3 smoothAcceleration;
		Vector3 angularVelocity;
		Vector3 smoothAngularVelocity;
		Vector3 smoothAngularAcceleration;

		Vector3 originalPosition;
		EulerAngle originalOrientation;
		float originalFov = 0.0f;
	};
}
