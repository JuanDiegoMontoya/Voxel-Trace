#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "vendor/helper_cuda.h"
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
__device__ __host__ void swap(T& a, T& b)
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