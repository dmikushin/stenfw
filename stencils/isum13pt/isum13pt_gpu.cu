/*
 * STENFW – a stencil framework for compilers benchmarking.
 *
 * Copyright (C) 2012 Dmitry Mikushin, University of Lugano
 *
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "isum13pt.h"

#include <stdio.h>

__global__ static void isum13pt_gpu_kernel(int nx, int ny, int ns,
	integer* w0, integer* w1, integer* w2)
{
	int i = blockIdx.x, j = blockIdx.y, k = threadIdx.x;
	
#define IDX(oi, oj, ok) (i + oi + 2 + (j + oj + 2) * nx + (k + ok + 2) * nx * ny)

	w2[IDX(0, 0, 0)] =  w1[IDX(0, 0, 0)] + w0[IDX(0, 0, 0)] +

			w0[IDX(+1, 0, 0)] + w0[IDX(-1, 0, 0)]  +
			w0[IDX(0, +1, 0)] + w0[IDX(0, -1, 0)]  +
			w0[IDX(0, 0, +1)] + w0[IDX(0, 0, -1)]  +

			w0[IDX(+2, 0, 0)] + w0[IDX(-2, 0, 0)]  +
			w0[IDX(0, +2, 0)] + w0[IDX(0, -2, 0)]  +
			w0[IDX(0, 0, +2)] + w0[IDX(0, 0, -2)]  +

			w1[IDX(+1, 0, 0)] + w1[IDX(-1, 0, 0)]  +
			w1[IDX(0, +1, 0)] + w1[IDX(0, -1, 0)]  +
			w1[IDX(0, 0, +1)] + w1[IDX(0, 0, -1)]  +

			w1[IDX(+2, 0, 0)] + w1[IDX(-2, 0, 0)]  +
			w1[IDX(0, +2, 0)] + w1[IDX(0, -2, 0)]  +
			w1[IDX(0, 0, +2)] + w1[IDX(0, 0, -2)];
}

#define CUDA_SAFE_CALL(call) {							\
	cudaError_t err = call;							\
	if( cudaSuccess != err) {						\
		fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",	\
			__FILE__, __LINE__, cudaGetErrorString(err) );		\
		exit(EXIT_FAILURE);						\
	}									\
}

extern "C" int isum13pt_gpu(int nx, int ny, int ns,
	integer* w0, integer* w1, integer* w2)
{
	isum13pt_gpu_kernel<<<dim3(nx - 4, ny - 4, 1), ns - 4>>>(nx, ny, ns, w0, w1, w2);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	return 0;
}

