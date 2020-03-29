#pragma once

__global__ void epicRayTracer(Voxels::Block* pWorld, glm::ivec3 worldDim,
	PerspectiveRayCamera camera, int numShadowRays, glm::vec2 imgSize,
	glm::vec3 chunkDim, Voxels::Light sun, curandState_t* states);

__host__ surface<void, 2>& GetScreenSurface();

struct ContextInfo
{
	__device__ __host__
	ContextInfo(Voxels::Block* pW, glm::vec3 wD,
		int nShw, Voxels::Light su, curandState_t& st,
		Ray ra) 
		: pWorld(pW), worldDim(wD), 
		numShadowRays(nShw), sun(su), 
		state(st), ray(ra) {}

	Voxels::Block* pWorld;
	glm::vec3 worldDim;
	int numShadowRays;
	Voxels::Light sun;
	curandState_t& state;
	Ray ray;
};