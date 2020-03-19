#include "stdafx.h"
#include "RayCamera.h"

PerspectiveRayCamera::PerspectiveRayCamera(
	glm::vec3 pos,
	glm::vec3 target,
	glm::vec3 upg,
	float fov,
	float aspectRatio)
{
	position = pos;
	forward = glm::normalize(target - position);
	right = glm::normalize(glm::cross(forward, upg));
	up = glm::cross(right, forward);

	dim.y = glm::tan(fov / 1);
	dim.x = dim.y * aspectRatio;
}

Ray PerspectiveRayCamera::makeRay(glm::vec2 p) const
{
	glm::vec3 dir = forward + p.x * dim.x * right + p.y * dim.y * up;
	return Ray(position, glm::normalize(dir));
}
