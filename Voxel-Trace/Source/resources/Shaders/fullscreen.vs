#version 450 core
layout (location = 0) in vec2 aPos;
layout (location = 0) in vec2 aTexCoord;

out vec2 vTexCoord;

void main()
{
	vTexCoord = aTexCoord;
	gl_Position = vec4(aPos, 0.0, 1.0);
}