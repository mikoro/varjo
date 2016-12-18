// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include <GL/gl3w.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>

#include "Film.h"
#include "Utils/GLUtils.h"
#include "Utils/CudaUtils.h"
#include "Utils/App.h"

using namespace Varjo;

void Film::initialize()
{
	App::getLog().logInfo("Initializing film");

	glGenTextures(1, &textureId);

	GLUtils::checkError("Could not create OpenGL texture");

	glBindTexture(GL_TEXTURE_2D, textureId);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glBindTexture(GL_TEXTURE_2D, 0);

	GLUtils::checkError("Could not set OpenGL texture parameters");

	programId = GLUtils::buildProgram("data/shaders/film.vert", "data/shaders/film.frag");

	textureUniformId = glGetUniformLocation(programId, "tex0");
	textureWidthUniformId = glGetUniformLocation(programId, "textureWidth");
	textureHeightUniformId = glGetUniformLocation(programId, "textureHeight");
	texelWidthUniformId = glGetUniformLocation(programId, "texelWidth");
	texelHeightUniformId = glGetUniformLocation(programId, "texelHeight");

	GLUtils::checkError("Could not get GLSL uniforms");

	const GLfloat vertexData[] =
	{
		-1.0f, -1.0f, 0.0f, 0.0f,
		1.0f, -1.0f, 1.0f, 0.0f,
		1.0f, 1.0f, 1.0f, 1.0f,
		-1.0f, -1.0f, 0.0f, 0.0f,
		1.0f, 1.0f, 1.0f, 1.0f,
		-1.0f, 1.0f, 0.0f, 1.0f,
	};

	glGenBuffers(1, &vboId);
	glBindBuffer(GL_ARRAY_BUFFER, vboId);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertexData), vertexData, GL_STATIC_DRAW);

	glGenVertexArrays(1, &vaoId);
	glBindVertexArray(vaoId);
	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), nullptr);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), reinterpret_cast<void*>(2 * sizeof(GLfloat)));
	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	GLUtils::checkError("Could not set OpenGL buffer parameters");
}

void Film::shutdown()
{
	if (texture != nullptr)
	{
		CudaUtils::checkError(cudaGraphicsUnregisterResource(texture), "Could not unregister OpenGL texture for CUDA");
		texture = nullptr;
	}

	glDeleteTextures(1, &textureId);

	GLUtils::checkError("Could not delete OpenGL texture");
}

void Film::resize(uint32_t width_, uint32_t height_)
{
	width = width_;
	height = height_;
	length = width * height;

	App::getLog().logInfo("Resizing film to %sx%s", width, height);

	if (texture != nullptr)
	{
		CudaUtils::checkError(cudaGraphicsUnregisterResource(texture), "Could not unregister OpenGL texture for CUDA");
		texture = nullptr;
	}

	glBindTexture(GL_TEXTURE_2D, textureId);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, GLsizei(width), GLsizei(height), 0, GL_RGBA, GL_FLOAT, nullptr);
	glBindTexture(GL_TEXTURE_2D, 0);

	GLUtils::checkError("Could not reserve OpenGL texture memory");

	CudaUtils::checkError(cudaGraphicsGLRegisterImage(&texture, textureId, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore), "Could not register OpenGL texture for CUDA");
}

void Film::render()
{
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, textureId);

	glUseProgram(programId);
	glUniform1i(textureUniformId, 0);
	glUniform1f(textureWidthUniformId, float(width));
	glUniform1f(textureHeightUniformId, float(height));
	glUniform1f(texelWidthUniformId, 1.0f / width);
	glUniform1f(texelHeightUniformId, 1.0f / height);

	glBindVertexArray(vaoId);
	glDrawArrays(GL_TRIANGLES, 0, 6);

	glBindVertexArray(0);
	glBindTexture(GL_TEXTURE_2D, 0);
	glUseProgram(0);

	GLUtils::checkError("Could not render the film");
}

uint32_t Film::getWidth() const
{
	return width;
}

uint32_t Film::getHeight() const
{
	return height;
}

uint32_t Film::getLength() const
{
	return length;
}

cudaGraphicsResource* Film::getTexture() const
{
	return texture;
}
