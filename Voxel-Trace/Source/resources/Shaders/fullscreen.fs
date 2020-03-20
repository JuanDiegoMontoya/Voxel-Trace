#version 450 core
out vec4 FragColor;

in vec2 vTexCoord;

uniform sampler2D tex;

void main()
{
	//vec2 size = textureSize(tex, 0);
	//FragColor = vec4(vTexCoord / size, 0., 1.);
	//FragColor = vec4(vTexCoord, 0., 1.);
	FragColor = vec4(texture(tex, vTexCoord).rgb, 1.0);
}