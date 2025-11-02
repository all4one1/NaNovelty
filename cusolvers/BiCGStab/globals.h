#pragma once
#include "device_launch_parameters.h"

namespace cusolver
{
	extern __device__ double rs_old;
	extern __device__ double rs_new;
	extern __device__ double alpha;
	extern __device__ double beta;
	extern __device__ double buffer;
	extern __device__ double buffer2;
	extern __device__ double omega;
}