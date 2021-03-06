#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "vendor/helper_cuda.h"
#include <math.h>
#include <curand.h>
#include <curand_kernel.h>
#define cudaCheck(err) checkCudaErrors(err, __FILE__, __LINE__)

template<int X, int Y>
__device__ __host__ glm::ivec3 expand(unsigned index)
{
	int k = index / (X * Y);
	int j = (index % (X * Y)) / X;
	int i = index - j * X - k * X * Y;
	return { i, j, k };
}

__device__ __host__
inline glm::ivec3 expand(unsigned index, int X, int Y)
{
	int k = index / (X * Y);
	int j = (index % (X * Y)) / X;
	int i = index - j * X - k * X * Y;
	return { i, j, k };
}

template<int X>
__device__ __host__ glm::ivec2 expand(unsigned index)
{
	return { index / X, index % X };
}

template<int X, int Y>
__device__ __host__ int flatten(glm::ivec3 coord)
{
	return coord.x + coord.y * X + coord.z * X * Y;
}

inline __device__ __host__ int flatten(glm::ivec3 coord, int X, int Y)
{
	return coord.x + coord.y * X + coord.z * X * Y;
}

template<int X>
__device__ __host__ int flatten(glm::ivec2 coord)
{
	return coord.x + coord.y * X;
}

template<int X, int Y, int Z>
__device__ __host__ bool inBoundary(glm::ivec3 p)
{
	return
		p.x >= 0 && p.x < X &&
		p.y >= 0 && p.y < Y &&
		p.z >= 0 && p.z < Z;
}

__device__
inline bool inBound(int a, int b)
{
	return a >= 0 && a < b;
}

template<typename T>
__device__ __host__ void cuswap(T& a, T& b)
{
	T tmp = std::move(a);
	a = std::move(b);
	b = std::move(tmp);
}

inline __device__ __host__ int flatten(glm::ivec2 coord, int X)
{
	return coord.x + coord.y * X;
}

inline __device__ __host__ glm::ivec2 expand(unsigned index, int Y)
{
	return { index / Y, index % Y };
}

inline __device__ __host__ 
float FresnelSchlick(glm::vec3 i, glm::vec3 n, float Eta, float Power)
{
	float F = ((1.0 - Eta) * (1.0 - Eta)) / ((1.0 + Eta) * (1.0 + Eta));
	return F + (1.0 - F) * glm::pow((1.0 - glm::dot(-i, n)), Power);
}

// returns an arbitrary vector that is orthogonal to the input
inline __device__ __host__
glm::vec3 OrthoVec(const glm::vec3& a)
{
	// a: choose any component
	// b: choose nonzero component
	// c: swap chosen components, negating one
	// d: set non-swapped component to 0
	// "smart" (dumb) way of doing this
	//cuswap(a[0], (a[1] == 0 ? (a[1]=0, a[2]) : (a[2]=0, a[1])));
	if (a[1] == 0) [[unlikely]] // actually nearly impossible, may remove
	{
		return { -a[2], 0, a[0] };
	}
	return { -a[1], a[0], 0 };
}

// expects normalized input!
// radius = half width of base of 1 unit high cone
// angle = slope of cone (radians)
inline __device__
glm::vec3 RandVecInCone(glm::vec3 dir, float angle, curandState_t& state)
{
	// generate random point on unit sphere
	// https://demonstrations.wolfram.com/RandomPointsOnASphere/
	float theta = curand_uniform(&state) * 2.f * glm::pi<float>(); // range 0 to 2pi
	float u = curand_uniform(&state) * 2.f - 1.f; // range -1 to 1
	float squ = glm::sqrt(1 - u * u); // avoid computing this twice
	glm::vec3 offset;
	offset.x = glm::cos(theta) * squ;
	offset.y = glm::sin(theta) * squ;
	offset.z = u;

	// height of triangle, from tan(y/x)=angle
	// x = 1 since this is unit cone
	float radius = glm::atan(angle);
	return dir + (offset * radius);
}