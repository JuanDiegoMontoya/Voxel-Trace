#include "stdafx.h"
#include "CommonDevice.cuh"
#include <stdio.h>
//#include <curand.h>
//#include <curand_kernel.h>
#include <math.h>

//#define N 25

#define MAX 100

/* this GPU kernel function is used to initialize the random states */
__global__ void init(unsigned int seed, curandState_t* states, int n)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	//printf("index = %d, stride = %d, n = %d\n", index, stride, n);
	for (int i = index; i < n; i += stride)
	{
		/* we have to initialize the state */
		curand_init(seed, /* the seed can be the same for each core, here we pass the time in from the CPU */
			i, /* the sequence number should be different for each core (unless you want all
										 cores to get the same sequence of numbers for some reason - use thread id! */
			0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
			&states[i]);
		//printf("%d\n", curand(&states[blockIdx.x * blockDim.x + threadIdx.x]));
	}
}

__global__ void randoms(curandState_t* states)
{
	/* curand works like rand - except that it takes a state as a parameter */
	printf("%d\n", curand(&states[blockIdx.x * blockDim.x + threadIdx.x]) % 100);
}


void InitCUDARand(curandState_t*& states, unsigned N)
{
	PERF_BENCHMARK_START;
	printf("Allocating %d rand states.\n", N);

	/* CUDA's random number library uses curandState_t to keep track of the seed value
		 we will store a random state for every thread  */
	//curandState_t* states;

	/* allocate space on the GPU for the random states */
	cudaMalloc((void**)&states, N * sizeof(curandState_t));

	const int KernelBlockSize = 64;
	const int KernelNumBlocks = (N + KernelBlockSize - 1) / KernelBlockSize;
	/* invoke the GPU to initialize all of the random states */
	init<<<KernelNumBlocks, KernelBlockSize>>>(time(0), states, N);
	cudaDeviceSynchronize();

	// test
	//glm::vec2 screenDim = { 500, 265 };
	//const int KernelBlockSize = 256;
	//const int KernelNumBlocks = (screenDim.x * screenDim.y + KernelBlockSize - 1) / KernelBlockSize;
	//randoms<<<KernelNumBlocks, KernelBlockSize>>>(states);
	//cudaDeviceSynchronize();
	PERF_BENCHMARK_END;
	printf("Took %.0fms", benchmark_duration_.count() * 1000.f);
}

void ShutdownCUDARands(curandState_t*& states)
{
	cudaFree(states);
}