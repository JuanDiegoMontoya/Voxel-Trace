#pragma once

struct Ray
{
	Ray(glm::vec3 o, glm::vec3 dir) : origin(o), direction(dir) {}
	glm::vec3 origin;
	glm::vec3 direction;
};

class RayCamera
{
public:
	virtual Ray makeRay(glm::vec2 p) const = 0;

};

class PerspectiveRayCamera : public RayCamera
{
public:
	PerspectiveRayCamera()
		: position(glm::vec3(0)),
		forward(glm::vec3(1, 0, 0)),
		right(glm::vec3(0, 0, 1)),
		up(glm::vec3(0, 1, 0)),
		dim(glm::vec2(0, 0)) {}
	PerspectiveRayCamera(
		glm::vec3 o, 
		glm::vec3 target,
		glm::vec3 upg,
		float fov,
		float aspectRatio);

	Ray makeRay(glm::vec2 p) const override;

private:
	glm::vec3 position;
	glm::vec3 forward;
	glm::vec3 right;
	glm::vec3 up;
	glm::vec2 dim; // w and h
};