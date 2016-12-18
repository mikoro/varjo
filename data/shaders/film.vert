#version 330

layout(location = 0) in vec2 position;
layout(location = 1) in vec2 texcoord;

out vec2 texcoordVarying;

void main()
{
	texcoordVarying = texcoord;
	gl_Position = vec4(position, 0.0f, 1.0f);
}
