#pragma once

namespace Voxels
{
	void Init();
	void InitGLStuff();
	void Render();

	void CameraRaySnapshot();

	struct Block
	{
		glm::vec3 diffuse;
		float alpha;
		float n; // index of refraction
	};
}