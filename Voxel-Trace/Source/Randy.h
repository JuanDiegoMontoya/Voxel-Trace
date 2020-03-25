#pragma once

#include <curand.h>
#include <curand_kernel.h>
void InitCUDARand(curandState_t*& states, unsigned numStates);

void ShutdownCUDARands(curandState_t*& states);