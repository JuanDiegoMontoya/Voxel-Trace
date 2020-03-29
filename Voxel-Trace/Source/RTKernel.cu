#include "stdafx.h"

#include "CommonDevice.cuh"
#include "Voxtrace.h"
#include "RayCamera.h"
#include "RTKernel.cuh"
#include "pick.h"

surface<void, 2> screenSurface;

__host__ surface<void, 2>& GetScreenSurface()
{
	return screenSurface;
}

struct PrimaryRayCaster
{
	__device__
	PrimaryRayCaster(ContextInfo& inf, glm::vec4& v) : info(inf), val(v) {}
	ContextInfo info;
	glm::vec4& val;
	int depthRemaining = 3;

	__device__
	bool operator()(
		glm::vec3 p, 
		Voxels::Block* block, 
		glm::vec3 norm, 
		glm::vec3 ex)
	{
		if (depthRemaining <= 0)
			return true;
		if (block)
		{
			if (block->alpha == 0)
				return false;

			// reflects, i am sorry
			bool refracted = false; // jank
			glm::vec3 refClr(block->diffuse);
			if (block->alpha < 1)
			{
				glm::vec3 reflDir = glm::normalize(glm::reflect(info.ray.direction, norm));
				PrimaryRayCaster castor = *this;
				castor.depthRemaining--;
				raycastBranchless(info.pWorld, info.worldDim, ex + reflDir * .001f, reflDir, 50.f, castor);
				return true; // uncomment when recursion is allowed
				//refClr = refrDir * .5f + .5f;
			}

			float visibility = 1;
			//int numShadowRays = numShadowRays;
			auto shadowCB = [&visibility, this](
				glm::vec3 p, Voxels::Block* block, glm::vec3 norm, glm::vec3)->bool
			{
				if (block && block->alpha == 1)
				{
					visibility -= 1.f / info.numShadowRays;
					return true;
				}
				return false;
			};

			glm::vec3 sunRay = glm::normalize(info.sun.position - ex); // block-to-light ray
			//raycastBranchless(pWorld, worldDim, ex + .02f * sunRay,
			//	sunRay, glm::min(glm::distance(sun.position, ex), 200.f), shadowCB);

			for (int i = 0; i < info.numShadowRays; i++)
			{
				float distToSun = glm::distance(info.sun.position, ex);
				float angle = glm::atan(info.sun.radius / distToSun);
				glm::vec3 shadowDir = RandVecInCone(info.sun.position - ex, angle, info.state);
				raycastBranchless(info.pWorld, info.worldDim, ex + .001f * shadowDir,
					shadowDir, glm::min(distToSun, 50.f), shadowCB);
				//block->diffuse = glm::vec3(angle * .5f + .5f);
			}

			//block->diffuse = shadowDir * .5f + .5f;
			//block->diffuse = ex / 2.f;

			// phong
			float diff = glm::max(glm::dot(sunRay, norm), 0.f);
			float spec = glm::pow(glm::max(glm::dot(info.ray.direction,
				glm::reflect(sunRay, norm)), 0.0f), 64.f);
			glm::vec3 ambient = glm::vec3(.2) * block->diffuse;
			glm::vec3 specular = glm::vec3(.7) * spec;
			glm::vec3 diffuse = block->diffuse * diff;

			diffuse *= visibility;
			specular *= visibility;

			// final color of pixel
			glm::vec3 FragColor(0);
			FragColor = diffuse + ambient + specular;
			val = glm::vec4(FragColor, 1.f);
			return true;
		}
		return false;
	};
};



__global__ void epicRayTracer(Voxels::Block* pWorld, glm::ivec3 worldDim,
	PerspectiveRayCamera camera, int numShadowRays, glm::vec2 imgSize,
	glm::vec3 chunkDim, Voxels::Light sun, curandState_t* states)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	int n = imgSize.x * imgSize.y;

	//printf("index = %d, stride = %d, n = %d\n", index, stride, n);
	for (int i = index; i < n; i += stride)
	{
		glm::vec2 imgPos = expand(i, imgSize.y);
		//glm::vec2 imgPos(x, y);
		glm::vec2 screenCoord(
			(2.0f * imgPos.x) / imgSize.x - 1.0f,
			(-2.0f * imgPos.y) / imgSize.y + 1.0f);
		Ray ray = camera.makeRay(screenCoord);


		// TODO: move all this into its own function so it can call itself recursively, etc

		glm::vec4 val{ .53f, .81f, .92f, 1 };
		PrimaryRayCaster primRay(
			ContextInfo(pWorld, worldDim, numShadowRays, sun, states[index], ray), val);
		raycastBranchless(pWorld, worldDim, ray.origin, ray.direction, 50, primRay);

		// write final pixel value
		surf2Dwrite(val, screenSurface, imgPos.x * sizeof(val), imgSize.y - imgPos.y - 1);
	}
}