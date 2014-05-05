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

#ifndef TEST_INIT_H
#define TEST_INIT_H

#ifdef CUDA
#include <cuda_runtime.h>
#endif
#ifdef MPI
#include <mpi.h>
#endif
#include <stddef.h>
#include <stdlib.h>

#include "config.h"
#include "grid.h"

#ifdef CUDA
#define CUDA_SAFE_CALL(call) {							\
	cudaError_t err = call;							\
	if( cudaSuccess != err) {						\
		fprintf(stderr, "CUDA error at %s:%i : %s\n",			\
			__FILE__, __LINE__, cudaGetErrorString(err));		\
		abort();							\
	}									\
}
#endif

#define MPI_ERR_STR_SIZE 1024

#ifdef MPI
#define MPI_ROOT_NODE 0
#define MPI_SAFE_CALL(call) {							\
	int err = call;								\
	if (err != MPI_SUCCESS) {						\
		char errstr[MPI_MAX_ERROR_STRING];				\
		int szerrstr;							\
		MPI_Error_string(err, errstr, &szerrstr);			\
		fprintf(stderr, "MPI error at %s:%i : %s\n",			\
			__FILE__, __LINE__, errstr);				\
		abort();							\
	}									\
}
#endif

// Recompute 3d grid index into 1d MPI node rank.
#define grid_rank1d(parent, grid) \
	(grid->ix + grid->iy * parent->sx + grid->is * parent->sx * parent->sy)

// Defines test configuration parameters.
struct test_config_t
{
	const char *name, *mode;

	// The local CPU grid domain.
	struct grid_domain_t cpu;
#ifdef CUDA
	// The local GPU grid domain.
	struct grid_domain_t gpu;
#endif
	int nx, ny, ns, nt;
	real dt;
	real c0, c1, c2;
	real dx, dy, ds;
	
	size_t size;
#ifdef MPI
	// The configuration of all decomposition grid domains,
	// and the local domain of entire MPI node.
	struct grid_domain_t* domains;
	int rank;
#endif
};

// Initialize a test instance on the given grid.
struct test_config_t* test_init(
	const char* name, const char* mode,
	int n, int nt, int sx, int sy, int ss, int rank, int szcomm,
	real xmin, real ymin, real zmin,
	real xmax, real ymax, real zmax,
	int bx, int by, int bs, int ex, int ey, int es
#ifdef CUDA
	, struct cudaDeviceProp* props
#endif
	);

// Dispose the test instance.
void test_dispose(struct test_config_t* t);

// Parse the test application command line.
void test_parse(int argc, char* argv[],
	const char** name_, const char** mode_, int* n_, int* nt_,
	int* sx_, int* sy_, int* ss_, int* rank_, int* szcomm_
#ifdef CUDA
	, struct cudaDeviceProp* props
#endif
	);

// Distribute the specified initial data across MPI nodes.
void test_load(struct test_config_t* t, int n, int sx, int sy, int ss,
	int szelem, char* data);

#endif // TEST_INIT_H

