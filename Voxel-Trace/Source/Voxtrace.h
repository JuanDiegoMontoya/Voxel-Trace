#pragma once

namespace Voxels
{
	void Init();
	void InitGLStuff();
	void InitBlocks();
	void Render();

	void CameraRaySnapshot();

	struct Block
	{
		Block() = default;
		Block(glm::vec3 d, float a, float in)
			: diffuse(d), alpha(a), n(in) {}
		glm::vec3 diffuse = { 1, 1, 1 };
		float alpha = 0;
		float n = 1.5; // index of refraction
	};
}