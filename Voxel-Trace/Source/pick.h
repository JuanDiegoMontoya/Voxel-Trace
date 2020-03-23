#pragma once

//#ifdef __CUDACC__
//__device__
//#endif
//void raycast(Voxels::Block* pWorld, glm::vec3 worldDim,
//	glm::vec3 origin, glm::vec3 direction, float radius,
//	bool callback(glm::vec3, Voxels::Block*, glm::vec3));

#ifdef __CUDACC__
__device__
#endif
float mod(float value, float modulus)
{
	return fmod((fmod(value, modulus)) + modulus, modulus);
}

#ifdef __CUDACC__
__device__
#endif
float intbound(float s, float ds)
{
	// Find the smallest positive t such that s+t*ds is an integer.
	if (ds < 0)
	{
		return intbound(-s, -ds);
	}
	else
	{
		s = mod(s, 1);
		// problem is now s+t*ds = 1
		return (1 - s) / ds;
	}
}

#ifdef __CUDACC__
__device__
#endif
glm::vec3 intbound(glm::vec3 s, glm::vec3 ds)
{
	return { intbound(s.x, ds.x), intbound(s.y, ds.y), intbound(s.z, ds.z) };
}

#ifdef __CUDACC__
__device__
#endif
int signum(float x)
{
	return x > 0 ? 1 : x < 0 ? -1 : 0;
}

/**
 * Call the callback with (x,y,z,value,face) of all blocks along the line
 * segment from point 'origin' in vector direction 'direction' of length
 * 'radius'. 'radius' may be infinite.
 *
 * 'face' is the normal vector of the face of that block that was entered.
 * It should not be used after the callback returns.
 *
 * If the callback returns a true value, the traversal will be stopped.
 */
template<typename CALLBACK>
#ifdef __CUDACC__
__device__
#endif
void raycast(Voxels::Block* pWorld, glm::ivec3 worldDim,
	glm::vec3 origin, glm::vec3 direction,
	float radius,
	//bool callback(glm::vec3, Voxels::Block*, glm::vec3))
	CALLBACK callback)
{
	glm::ivec3 p = glm::floor(origin);
	glm::vec3 d = direction;
	glm::ivec3 step = glm::sign(d);
	glm::vec3 tMax = intbound(origin, d);
	glm::vec3 tDelta = glm::vec3(step) / d;
	glm::vec3 face(0);

	// Avoids an infinite loop.
	//ASSERT_MSG(d != glm::vec3(0), "Raycast in zero direction!");

	radius /= glm::length(d);

	while (1)
	{
		//printf("ray pos: %d, %d, %d\n", p.x, p.y, p.z);
		// use null block if not within bounds of world
		Voxels::Block* block;
		if (glm::any(glm::lessThan(p, glm::ivec3(0, 0, 0))) ||
			glm::any(glm::greaterThanEqual(p, worldDim)))
			block = nullptr;
		else // TODO: make sure this flatten function is correct by testing various input Xs and Ys
			block = &pWorld[flatten(p, worldDim.x, worldDim.y)];

		if (callback(p, block, face))
			break;

		// tMax.x stores the t-value at which we cross a cube boundary along the
		// X axis, and similarly for Y and Z. Therefore, choosing the least tMax
		// chooses the closest cube boundary. Only the first case of the four
		// has been commented in detail.
		if (tMax.x < tMax.y)
		{
			if (tMax.x < tMax.z)
			{
				if (tMax.x > radius) break;
				// Update which cube we are now in.
				p.x += step.x;
				// Adjust tMax.x to the next X-oriented boundary crossing.
				tMax.x += tDelta.x;
				// Record the normal vector of the cube face we entered.
				face.x = float(-step.x);
				face.y = 0;
				face.z = 0;
			}
			else
			{
				if (tMax.z > radius) break;
				p.z += step.z;
				tMax.z += tDelta.z;
				face.x = 0;
				face.y = 0;
				face.z = float(-step.z);
			}
		}
		else
		{
			if (tMax.y < tMax.z)
			{
				if (tMax.y > radius) break;
				p.y += step.y;
				tMax.y += tDelta.y;
				face.x = 0;
				face.y = float(-step.y);
				face.z = 0;
			}
			else
			{
				// Identical to the second case, repeated for simplicity in
				// the conditionals.
				if (tMax.z > radius) break;
				p.z += step.z;
				tMax.z += tDelta.z;
				face.x = 0;
				face.y = 0;
				face.z = float(-step.z);
			}
		}
	}
}

template<typename CALLBACK> 
#ifdef __CUDACC__
__device__
#endif
// modified from https://www.shadertoy.com/view/4dX3zl#
void raycastBranchless(Voxels::Block* pWorld, glm::ivec3 worldDim,
	glm::vec3 origin, glm::vec3 direction,
	float radius, CALLBACK callback)
{
	auto rayPos = origin;
	auto rayDir = direction;
	glm::ivec3 mapPos = glm::ivec3(glm::floor(rayPos + 0.f));
	glm::vec3 deltaDist = glm::abs(glm::vec3(glm::length(rayDir)) / rayDir);
	glm::ivec3 rayStep = glm::ivec3(sign(rayDir));
	glm::vec3 sideDist = (glm::sign(rayDir) * (glm::vec3(mapPos) - rayPos) + 
		(glm::sign(rayDir) * 0.5f) + 0.5f) * deltaDist;
	glm::bvec3 mask;
	glm::vec3 norm(0);
	glm::vec3 exact(sideDist);

	for (int i = 0; i < radius; i++)
	{
		Voxels::Block* block;
		if (glm::any(glm::lessThan(mapPos, glm::ivec3(0, 0, 0))) ||
			glm::any(glm::greaterThanEqual(mapPos, worldDim)))
			block = nullptr;
		else
			block = &pWorld[flatten(mapPos, worldDim.x, worldDim.y)];
		//if (getVoxel(mapPos))
			//continue;
		if (callback(mapPos, block, norm, exact))
			break;

		// find which component(s) are nearest to edge
		glm::vec3 a(sideDist.x, sideDist.y, sideDist.z);
		glm::vec3 b(sideDist.y, sideDist.z, sideDist.x);
		glm::vec3 c(sideDist.z, sideDist.x, sideDist.y);
		mask = glm::lessThanEqual(a, glm::min(b, c));

		// advance ray
		sideDist += glm::vec3(mask) * deltaDist;
		mapPos += norm = glm::ivec3(glm::vec3(mask)) * rayStep;
		norm *= -1; // norm is opposite of step

		// find the exact location the ray crossed block borders
		exact = glm::vec3(mapPos) - sideDist * glm::vec3(mask);
	}
}