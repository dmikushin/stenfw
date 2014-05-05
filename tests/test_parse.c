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

#include <stdio.h>
#include <string.h>

static void usage(const char* filename)
{
#if defined(MPI) && defined(CUDA)
	printf("Usage: %s <n> <nt> <sx> <sy> <ss> <CPU|GPU>\n", filename);
#elif defined(MPI)
	printf("Usage: %s <n> <nt> <sx> <sy> <ss>\n", filename);
#elif defined(CUDA)
	printf("Usage: %s <n> <nt> <CPU|GPU>\n", filename);
#else
	printf("Usage: %s <n> <nt>\n", filename);
#endif
	printf("where <n> and <nt> must be positive\n");
#if defined(MPI)
	printf("and <sx> * <sy> * <ss> must equal to the number of MPI processes\n");
#endif
	exit(0);
}

// Parse the test application command line.
void test_parse(int argc, char* argv[],
	const char** name_, const char** mode_, int* n_, int* nt_,
	int* sx_, int* sy_, int* ss_, int* rank_, int* szcomm_
#ifdef CUDA
	, struct cudaDeviceProp* props
#endif
	)
{
	const char* name = argv[0];

#if defined(MPI) && defined(CUDA)
	int nargs = 7;
#elif defined(MPI)
	int nargs = 6;
#elif defined(CUDA)
	int nargs = 4;
#else
	int nargs = 3;
#endif
	if (argc != nargs)
		usage(name);

	int n = atoi(argv[1]);
	int nt = atoi(argv[2]);
#ifndef MPI
	const char* mode = argv[3];
#else
	int sx = atoi(argv[3]);
	int sy = atoi(argv[4]);
	int ss = atoi(argv[5]);
	const char* mode = argv[6];
#endif

#if defined(MPI) && defined(CUDA)	
	if ((n <= 0) || (nt <= 0) || (sx <= 0) || (sy <= 0) || (ss <= 0) ||
		(strcmp(mode, "CPU") && strcmp(mode, "GPU")))
#elif defined(MPI)
	if ((n <= 0) || (nt <= 0) || (sx <= 0) || (sy <= 0) || (ss <= 0))
#elif defined(CUDA)
	if ((n <= 0) || (nt <= 0) || (strcmp(mode, "CPU") && strcmp(mode, "GPU")))
#else
	if ((n <= 0) || (nt <= 0))
#endif
		usage(name);

#ifdef MPI
	int mpi_initialized = 0;
	MPI_SAFE_CALL(MPI_Initialized(&mpi_initialized));
	if (!mpi_initialized)
		MPI_SAFE_CALL(MPI_Init(&argc, &argv));
	int rank, szcomm, sxys = sx * sy * ss;
	MPI_SAFE_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
	MPI_SAFE_CALL(MPI_Comm_size(MPI_COMM_WORLD, &szcomm));
	if (szcomm != sxys)
	{
		if (rank == MPI_ROOT_NODE)
			fprintf(stderr, "The value of <sx> * <sy> * <ss> must equal to the number of MPI processes.\n");
		MPI_SAFE_CALL(MPI_Finalize());
		exit(EXIT_FAILURE);
	}
#endif
	
#if defined(CUDA)
	if (!strcmp(mode, "GPU"))
	{
		int count = 0;
		CUDA_SAFE_CALL(cudaGetDeviceCount(&count));
		if (!count)
		{
			fprintf(stderr, "No CUDA-enabled devices found\n");
			exit(EXIT_FAILURE);
		}
		
#if defined(CUDA_MAPPED)
		// Enable device-mapped pinned memory.
		CUDA_SAFE_CALL(cudaGetDeviceProperties(props, 0));
		if (!props->canMapHostMemory)
		{
			fprintf(stderr, "Device 0 does not support host memory mapping\n");
			exit(EXIT_FAILURE);
		}
		CUDA_SAFE_CALL(cudaSetDeviceFlags(cudaDeviceMapHost));
#endif // CUDA_MAPPED

#ifdef MPI
		// TODO: Select GPU for each MPI process.
		// Algorithm: determine the host name of each MPI process,
		// query the number of available GPUs on each host and
		// exclusively assign GPUs to MPI processes.
		// Should also account PCI->socket bindings.
		// Use hwloc?
#endif
	}
#endif // CUDA

	*name_ = name;
	*mode_ = mode;
	*n_ = n; *nt_ = nt;
#ifdef MPI
	*sx_ = sx; *sy_ = sy; *ss_ = ss;
	*rank_ = rank; *szcomm_ = szcomm;
#else
	*sx_ = 1; *sy_ = 1; *ss_ = 1;
	*rank_ = 0; *szcomm_ = 1;
#endif // MPI
}

// Dispose the test instance.
void test_dispose(struct test_config_t* t)
{
	const char* mode = t->mode;
	int narrays = t->cpu.narrays;

	//
	// 4) Dispose the CPU data arrays.
	//
#if defined(CUDA)
	if (!strcmp(mode, "GPU"))
	{
		for (int iarray = 0; iarray < narrays; iarray++)
		{
#if defined(CUDA_MAPPED) || defined(CUDA_PINNED)
			// Allocate memory as host-mapped memory accessible both from
			// CPU and GPU.
			CUDA_SAFE_CALL(cudaFreeHost(t->cpu.arrays[iarray]));
#endif // CUDA_MAPPED || CUDA_PINNED
		}
	}
	else
#endif // CUDA
	{
		for (int iarray = 0; iarray < narrays; iarray++)
			free(t->cpu.arrays[iarray]);
	}

#if defined(MPI)

	// Dispose memory required to keep the rest of domain data.
#if defined(CUDA) &&  defined(CUDA_MAPPED)
	if (!strcmp(mode, "GPU"))
	{
		for (int i = 0; i < t->cpu.nsubdomains; i++)
		{
			struct grid_domain_t* subdomain = t->cpu.subdomains + i;	
			for (int iarray = 0; iarray < narrays; iarray++)
				CUDA_SAFE_CALL(cudaFreeHost(subdomain->arrays[iarray]));
			free(subdomain->arrays);
		}
	}
	else
#endif // CUDA && CUDA_MAPPED
	{
		for (int i = 0; i < t->cpu.nsubdomains; i++)
		{
			struct grid_domain_t* subdomain = t->cpu.subdomains + i;	
			for (int iarray = 0; iarray < narrays; iarray++)
				free(subdomain->arrays[iarray]);
			free(subdomain->arrays);
		}
	}
#endif // MPI
	
	//
	// 5) Dispose the GPU data arrays.
	//
#if defined(CUDA)
	if (!strcmp(mode, "GPU"))
	{
#if !defined(CUDA_MAPPED)
		for (int iarray = 0; iarray < narrays; iarray++)
			CUDA_SAFE_CALL(cudaFree(t->gpu.arrays[iarray]));
#endif // CUDA_MAPPED
	}
#endif // CUDA

#ifdef MPI
	free(t->domains);
#endif
	free(t);

#ifdef MPI
	int mpi_initialized = 0;
	MPI_SAFE_CALL(MPI_Initialized(&mpi_initialized));
	int mpi_finalized = 0;
	MPI_SAFE_CALL(MPI_Finalized(&mpi_finalized));
	if (mpi_initialized && !mpi_finalized)
		MPI_SAFE_CALL(MPI_Finalize());
#endif
}

