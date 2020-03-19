#version 450 core
out vec4 FragColor;

in vec2 vTexCoord;

uniform sampler2D tex;

void main()
{
	FragColor = vec4(texture(tex, vTexCoord).rgb, 1.0);
}