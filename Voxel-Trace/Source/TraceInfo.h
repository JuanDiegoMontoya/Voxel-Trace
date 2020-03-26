#pragma once
#include "RayCamera.h"

struct TraceInfo
{
	PerspectiveRayCamera camera;
	int numShadowRays;
	glm::vec2 imgSize;
};