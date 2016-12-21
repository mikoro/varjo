// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include <GLFW/glfw3.h>

#include "tinyformat/tinyformat.h"

#include "Utils/App.h"
#include "Utils/Log.h"
#include "Core/Camera.h"
#include "Math/MathUtils.h"

using namespace Varjo;

void Camera::initialize()
{
	originalPosition = position;
	originalOrientation = orientation;
	originalFov = fov;
}

void Camera::setFilmSize(uint32_t width, uint32_t height)
{
	filmWidth = float(width);
	filmHeight = float(height);
}

void Camera::reset()
{
	position = originalPosition;
	orientation = originalOrientation;
	fov = originalFov;
	currentSpeedModifier = 1.0f;
	velocity = Vector3(0.0f, 0.0f, 0.0f);
	smoothVelocity = Vector3(0.0f, 0.0f, 0.0f);
	smoothAcceleration = Vector3(0.0f, 0.0f, 0.0f);
	angularVelocity = Vector3(0.0f, 0.0f, 0.0f);
	smoothAngularVelocity = Vector3(0.0f, 0.0f, 0.0f);
	smoothAngularAcceleration = Vector3(0.0f, 0.0f, 0.0f);
}

void Camera::update(float timeStep)
{
	Window& window = App::getWindow();
	MouseInfo mouseInfo = window.getMouseInfo();

	// SPEED MODIFIERS //

	if (window.keyWasPressed(GLFW_KEY_INSERT))
		currentSpeedModifier *= 2.0f;

	if (window.keyWasPressed(GLFW_KEY_DELETE))
		currentSpeedModifier *= 0.5f;

	float actualMoveSpeed = moveSpeed * currentSpeedModifier;

	if (window.keyIsDown(GLFW_KEY_LEFT_CONTROL) || window.keyIsDown(GLFW_KEY_RIGHT_CONTROL))
		actualMoveSpeed *= slowSpeedModifier;

	if (window.keyIsDown(GLFW_KEY_LEFT_SHIFT) || window.keyIsDown(GLFW_KEY_RIGHT_SHIFT))
		actualMoveSpeed *= fastSpeedModifier;

	if (window.keyIsDown(GLFW_KEY_LEFT_ALT) || window.keyIsDown(GLFW_KEY_RIGHT_ALT))
		actualMoveSpeed *= veryFastSpeedModifier;

	// ACCELERATIONS AND VELOCITIES //

	velocity = Vector3(0.0f, 0.0f, 0.0f);
	angularVelocity = Vector3(0.0f, 0.0f, 0.0f);
	bool movementKeyIsPressed = false;

	if (window.mouseIsDown(GLFW_MOUSE_BUTTON_LEFT) || freeLook)
	{
		smoothAngularAcceleration.y -= mouseInfo.deltaX * mouseSpeed;
		smoothAngularAcceleration.x += mouseInfo.deltaY * mouseSpeed;
		angularVelocity.y = -mouseInfo.deltaX * mouseSpeed;
		angularVelocity.x = mouseInfo.deltaY * mouseSpeed;
	}

	if (window.keyIsDown(GLFW_KEY_W))
	{
		smoothAcceleration += forward * actualMoveSpeed;
		velocity = forward * actualMoveSpeed;
		movementKeyIsPressed = true;
	}

	if (window.keyIsDown(GLFW_KEY_S))
	{
		smoothAcceleration -= forward * actualMoveSpeed;
		velocity = -forward * actualMoveSpeed;
		movementKeyIsPressed = true;
	}

	if (window.keyIsDown(GLFW_KEY_D))
	{
		smoothAcceleration += right * actualMoveSpeed;
		velocity = right * actualMoveSpeed;
		movementKeyIsPressed = true;
	}

	if (window.keyIsDown(GLFW_KEY_A))
	{
		smoothAcceleration -= right * actualMoveSpeed;
		velocity = -right * actualMoveSpeed;
		movementKeyIsPressed = true;
	}

	if (window.keyIsDown(GLFW_KEY_E))
	{
		smoothAcceleration += up * actualMoveSpeed;
		velocity = up * actualMoveSpeed;
		movementKeyIsPressed = true;
	}

	if (window.keyIsDown(GLFW_KEY_Q))
	{
		smoothAcceleration -= up * actualMoveSpeed;
		velocity = -up * actualMoveSpeed;
		movementKeyIsPressed = true;
	}

	if (window.keyIsDown(GLFW_KEY_SPACE) || !enableMovement)
	{
		velocity = Vector3(0.0f, 0.0f, 0.0f);
		smoothVelocity = Vector3(0.0f, 0.0f, 0.0f);
		smoothAcceleration = Vector3(0.0f, 0.0f, 0.0f);
		angularVelocity = Vector3(0.0f, 0.0f, 0.0f);
		smoothAngularVelocity = Vector3(0.0f, 0.0f, 0.0f);
		smoothAngularAcceleration = Vector3(0.0f, 0.0f, 0.0f);
	}

	// EULER INTEGRATION //

	cameraIsMoving = false;

	if (smoothMovement)
	{
		smoothVelocity += smoothAcceleration * timeStep;
		position += smoothVelocity * timeStep;

		smoothAngularVelocity += smoothAngularAcceleration * timeStep;
		orientation.yaw += smoothAngularVelocity.y * timeStep;
		orientation.pitch += smoothAngularVelocity.x * timeStep;

		cameraIsMoving = !smoothVelocity.isZero() || !smoothAngularVelocity.isZero();
	}
	else
	{
		position += velocity * timeStep;
		orientation.yaw += angularVelocity.y * timeStep;
		orientation.pitch += angularVelocity.x * timeStep;

		cameraIsMoving = !velocity.isZero() || !angularVelocity.isZero();
	}

	// DRAG & AUTO STOP //

	float smoothVelocityLength = smoothVelocity.length();
	float smoothAngularVelocityLength = smoothAngularVelocity.length();

	if ((smoothVelocityLength < autoStopSpeed * actualMoveSpeed) && !movementKeyIsPressed)
		smoothVelocity = smoothAcceleration = Vector3(0.0f, 0.0f, 0.0f);
	else if (!smoothVelocity.isZero())
		smoothAcceleration = moveDrag * (-smoothVelocity.normalized() * smoothVelocityLength);
	
	if (smoothAngularVelocityLength < autoStopSpeed * actualMoveSpeed)
		smoothAngularVelocity = smoothAngularAcceleration = Vector3(0.0f, 0.0f, 0.0f);
	else if (!smoothAngularVelocity.isZero())
		smoothAngularAcceleration = mouseDrag * (-smoothAngularVelocity.normalized() * smoothAngularVelocityLength);

	// ORIENTATION & BASIS VECTORS //

	orientation.clampPitch();
	orientation.normalize();
	forward = orientation.getDirection();
	right = forward.cross(Vector3::almostUp()).normalized();
	up = right.cross(forward).normalized();

	// MISC

	fov = MAX(1.0f, MIN(fov, 180.0f));
	float filmDistance = (filmWidth / 2.0f) / std::tan(MathUtils::degToRad(fov / 2.0f));
	filmCenter = position + (forward * filmDistance);
}

bool Camera::isMoving() const
{
	return cameraIsMoving;
}

void Camera::saveState(const std::string& fileName) const
{
	App::getLog().logInfo("Saving camera state to %s", fileName);

	std::ofstream file(fileName);

	if (!file.is_open())
		throw std::runtime_error("Could not open file for writing camera state");

	file << tfm::format("x = %f", position.x) << std::endl;
	file << tfm::format("y = %f", position.y) << std::endl;
	file << tfm::format("z = %f", position.z) << std::endl;
	file << std::endl;
	file << tfm::format("pitch = %f", orientation.pitch) << std::endl;
	file << tfm::format("yaw = %f", orientation.yaw) << std::endl;
	file << tfm::format("roll = %f", orientation.roll) << std::endl;
	file << std::endl;
	file << tfm::format("fov = %f", fov) << std::endl;
	file << std::endl;
	file << tfm::format("scene.camera.position = Vector3(%.4ff, %.4ff, %.4ff);", position.x, position.y, position.z) << std::endl;
	file << tfm::format("scene.camera.orientation = EulerAngle(%.4ff, %.4ff, %.4ff);", orientation.pitch, orientation.yaw, orientation.roll) << std::endl;
	file << tfm::format("scene.camera.fov = %.2ff;", fov) << std::endl;

	file.close();
}

CameraData Camera::getCameraData() const
{
	CameraData data;

	data.position = make_float3(position.x, position.y, position.z);
	data.right = make_float3(right.x, right.y, right.z);
	data.up = make_float3(up.x, up.y, up.z);
	data.forward = make_float3(forward.x, forward.y, forward.z);
	data.filmCenter = make_float3(filmCenter.x, filmCenter.y, filmCenter.z);
	data.halfFilmWidth = filmWidth / 2.0f;
	data.halfFilmHeight = filmHeight / 2.0f;

	return data;
}

Vector3 Camera::getRight() const
{
	return right;
}

Vector3 Camera::getUp() const
{
	return up;
}

Vector3 Camera::getForward() const
{
	return forward;
}
