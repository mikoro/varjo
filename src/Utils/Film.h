// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <cstdint>

#include <GL/glcorearb.h>
#include <cuda_runtime.h>

/*
The idea of the film is to reserve memory on the GPU as a OpenGL texture.
The texture can be smaller than the actual window size.
The the CUDA renderer will map the texture as a cudaArray and finally output its stuff to it.
The film will then render itself to the window, resampling the possibly smaller texture to window size.
 */

namespace Varjo
{
	class Film
	{
	public:

		void initialize();
		void shutdown();
		void resize(uint32_t width, uint32_t height);
		void render();

		uint32_t getWidth() const;
		uint32_t getHeight() const;
		uint32_t getLength() const;

		cudaGraphicsResource* getTexture() const;

	private:

		uint32_t width = 0;
		uint32_t height = 0;
		uint32_t length = 0;

		GLuint vaoId = 0;
		GLuint vboId = 0;
		GLuint programId = 0;
		GLuint textureId = 0;
		GLuint textureUniformId = 0;
		GLuint textureWidthUniformId = 0;
		GLuint textureHeightUniformId = 0;
		GLuint texelWidthUniformId = 0;
		GLuint texelHeightUniformId = 0;

		cudaGraphicsResource* texture = nullptr;
	};
}
