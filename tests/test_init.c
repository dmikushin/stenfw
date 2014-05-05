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

#include "test_init.h"

#include <assert.h>
#include <malloc.h>
#include <stdio.h>
#include <string.h>

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
)
{
	// TODO: replace n with nx, ny, ns.

	// TODO: parameterize.
	int szelem = sizeof(real);
	int narrays = 3;

	//
	// 1) Calculate the dimensions of entire grid domain.
	//
#ifdef MPI
	// For each MPI node create a view of decomposed grid topology.
	struct grid_domain_t* domains = grid_init_simple(
		n, n, n, sx, sy, ss, bx, by, bs, ex, ey, es);

	// The rank-th subdomain is assigned to entire MPI process.
	struct grid_domain_t* domain = domains + rank;

	// Set domain data copying callbacks and user-defined pointer
	// - the test config, in this case.
	int ndomains = domain->parent->nsubdomains;
	for (int i = 0; i < ndomains; i++)
	{
		struct grid_domain_t* domain = domains + i;
		domain->scatter_memcpy = &grid_subcpy;
		domain->gather_memcpy = &grid_subcpy;
		domain->narrays = narrays;
		domain->szelem = szelem;
	}

	// The problem X, Y, Z dimensions are set relative to the
	// subdomain of entire MPI process.
	int nx = domain->grid[0].nx, ny = domain->grid[0].ny, ns = domain->grid[0].ns;
	size_t nxys = domain->grid[0].extsize;
	size_t nxysb = nxys * szelem;
#else
	int nx = n, ny = n, ns = n;
	size_t nxys = nx * ny * ns;
	size_t nxysb = nxys * szelem;
#endif

	//
	// 2) Allocate the test config structure together with
	// the array of pointers to keep CPU and GPU data arrays.
	// Assign dimensions and data pointers.
	//
#ifdef CUDA
	int gpu = !strcmp(mode, "GPU");
#else
	int gpu = 0;
#endif
	struct test_config_t* t = (struct test_config_t*)malloc(
		sizeof(struct test_config_t) + (1 + gpu) * narrays * sizeof(char*));
#ifdef MPI
	t->cpu = *domain;
#ifdef CUDA
	t->gpu = *domain;
#endif
	// Track MPI node rank, and decomposition grid domains
	// in test config structure.
	t->rank = rank;
	t->domains = domains;
#else
	t->cpu.grid->nx = nx; t->cpu.grid->ny = ny; t->cpu.grid->ns = ns; t->cpu.grid->extsize = nxys;
	t->cpu.parent = &t->cpu;
	t->cpu.narrays = narrays;
#ifdef CUDA
	t->gpu.grid->nx = nx; t->gpu.grid->ny = ny; t->gpu.grid->ns = ns; t->gpu.grid->extsize = nxys;
	t->gpu.parent = &t->gpu;
	t->cpu.narrays = narrays;
#endif
#endif
	t->cpu.arrays = (char**)(t + 1);
#ifdef CUDA
	t->gpu.arrays = t->cpu.arrays + narrays;
#endif

	//
	// 3) Set the simple properties of test config.
	//
	t->name = name; t->mode = mode;
	t->nx = nx; t->ny = ny; t->ns = ns; t->nt = nt;

	// Grid steps.
	t->dx = (xmax - xmin) / (n - 1);
	t->dy = (ymax - ymin) / (n - 1);
	t->ds = (zmax - zmin) / (n - 1);
	t->dt = t->dx / 2.0;
		
	// Set scheme coefficients.
	double dt2dx2 = (t->dt * t->dt) / (t->dx * t->dx);
	t->c0 = 2.0 - dt2dx2 * 7.5;
	t->c1 = dt2dx2 * (4.0 / 3.0);
	t->c2 = dt2dx2 * (-1.0 / 12.0);

	//
	// 4) Allocate the CPU data arrays.
	//
#if defined(CUDA)
	if (!strcmp(mode, "GPU"))
	{
		for (int iarray = 0; iarray < narrays; iarray++)
		{
#if defined(CUDA_MAPPED)
			// Allocate memory as host-mapped memory accessible both from
			// CPU and GPU.
			CUDA_SAFE_CALL(cudaHostAlloc((void**)&t->cpu.arrays[iarray],
				nxysb, cudaHostAllocMapped));
#elif defined(CUDA_PINNED)
			// Allocate host memory as pinned to get faster CPU-GPU data
			// transfers.
			CUDA_SAFE_CALL(cudaMallocHost((void**)&t->cpu.arrays[iarray],
				nxysb));
#endif // CUDA_MAPPED
		}
	}
	else
#endif // CUDA
	{
		// Allocate regular CPU memory.
		for (int iarray = 0; iarray < narrays; iarray++)
			t->cpu.arrays[iarray] = (char*)malloc(nxysb);
	}

	// Initially flush CPU array data to zero.
	for (int iarray = 0; iarray < narrays; iarray++)
		memset(t->cpu.arrays[iarray], 0, nxysb);
#if defined(MPI)
	struct grid_domain_t* subdomains = domain->subdomains;
	int nsubdomains = domain->nsubdomains;

#if defined(CUDA) && !defined(CUDA_MAPPED)
	if (!strcmp(mode, "GPU"))
	{
		// Assign domain main arrays.
		domain->arrays = t->gpu.arrays;
	}
	else
#endif // CUDA && !CUDA_MAPPED
	{
		// Assign domain main arrays.
		domain->arrays = t->cpu.arrays;
	}

	// Allocate memory required to keep the rest of domain data.
	// In addition to main data arrays, each domain also allocates data
	// for its subdomains (nested domains). In this case the nested domains
	// represent boundaries for data buffering.
#if defined(CUDA) && defined(CUDA_MAPPED)
	if (!strcmp(mode, "GPU"))
	{
		for (int i = 0; i < nsubdomains; i++)
		{
			struct grid_domain_t* subdomain = subdomains + i;
	
			subdomain->arrays = (char**)malloc(sizeof(char*) * narrays);
			subdomain->narrays = narrays;
			for (int iarray = 0; iarray < narrays; iarray++)
			{
				size_t size = subdomain->grid[0].extsize * szelem;

				// Allocate a host-mapped array for subdomain in order
				// to make in possible to perform GPU-initiated boundaries
				// update.
				CUDA_SAFE_CALL(cudaHostAlloc((void**)&subdomain->arrays[iarray],
					size, cudaHostAllocMapped));

				// TODO: mapping
				
				// TODO: flushing to zero.
			}
		}
	}
	else
#endif // CUDA && CUDA_MAPPED
	{
		for (int i = 0; i < nsubdomains; i++)
		{
			struct grid_domain_t* subdomain = subdomains + i;
	
			subdomain->arrays = (char**)malloc(sizeof(char*) * narrays);
			subdomain->narrays = narrays;
			for (int iarray = 0; iarray < narrays; iarray++)
			{
				size_t size = subdomain->grid[0].extsize * szelem;

				// Allocate regular CPU memory.
				subdomain->arrays[iarray] = (char*)malloc(size);
				
				// Flush to zero.
				memset(subdomain->arrays[iarray], 0, size);
			}
		}
	}
#endif // MPI
	
	//
	// 5) Allocate the GPU data arrays.
	//
#if defined(CUDA)
	if (!strcmp(mode, "GPU"))
	{
#if defined(CUDA_MAPPED)
		// In case of host-mapped memory the GPU arrays pointers are
		// either same as for CPU arrays or contain specially mapped
		// pointers, depending on device capability.
		int use_mapping = props->major < 2;
		if (use_mapping)
		{
#ifdef VERBOSE
			printf("requires mapping\n");
#endif
			for (int i = 0; i < narrays; i++)
				CUDA_SAFE_CALL(cudaHostGetDevicePointer(
					(void**)&t->gpu.arrays[i], t->cpu.arrays[i], 0));
		}
		else
		{
#ifdef VERBOSE
			printf("does not require mapping\n");
#endif
			for (int iarray = 0; iarray < narrays; iarray++)
				t->gpu.arrays[iarray] = t->cpu.arrays[iarray];
		}
#else
		for (int iarray = 0; iarray < narrays; iarray++)
		{
			// Allocate regular GPU memory.
			CUDA_SAFE_CALL(cudaMalloc((void**)&t->gpu.arrays[iarray], nxysb));

			// Initially flush GPU array data to zero.
			CUDA_SAFE_CALL(cudaMemset(t->gpu.arrays[iarray], 0, nxysb));
			
			// TODO: reassign arrays of MPI domain.
		}
#endif // CUDA_MAPPED
	}
#endif // CUDA

	return t;
}

