#include "stdafx.h"
#include <functional>
#include "Voxtrace.h"
#include "pick.h"

float mod(float value, float modulus)
{
	return fmod((fmod(value, modulus)) + modulus, modulus);
}

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

glm::vec3 intbound(glm::vec3 s, glm::vec3 ds)
{
	return { intbound(s.x, ds.x), intbound(s.y, ds.y), intbound(s.z, ds.z) };
}

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
void raycast(glm::vec3 origin, glm::vec3 direction, float radius, std::function<bool(glm::vec3, Voxels::Block, glm::vec3) > callback)
{
	glm::ivec3 p = glm::floor(origin);
	glm::vec3 d = direction;
	glm::ivec3 step = glm::sign(d);
	glm::vec3 tMax = intbound(origin, d);
	glm::vec3 tDelta = glm::vec3(step) / d;
	glm::vec3 face(0);

	// Avoids an infinite loop.
	ASSERT_MSG(d != glm::vec3(0), "Raycast in zero direction!");

	radius /= glm::length(d);

	while (1)
	{
		// Invoke the callback, unless we are not *yet* within the bounds of the
		// world.
		if (callback(p, /*Chunk::AtWorld(p)*/ Voxels::Block(), face))
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