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

#include "genirand.h"
#include "isum13pt.h"
#include "isum13pt_init.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <timing.h>

// The problem physical positioning.
integer xmin = -1.0, ymin = -1.0, zmin = -1.0;
integer xmax =  1.0, ymax =  1.0, zmax =  1.0;

// The thicknesses of isum13pt stencil are 2 in all directions.
int bx = 2, by = 2, bs = 2, ex = 2, ey = 2, es = 2;

int main(int argc, char* argv[])
{
	// Parse test command line arguments, perform early
	// initializations.
	const char *name, *mode;
	int n, nt, sx, sy, ss, rank, szcomm;
#ifdef CUDA
	struct cudaDeviceProp props;
#endif
	test_parse(argc, argv, &name, &mode,
		&n, &nt, &sx, &sy, &ss, &rank, &szcomm
#ifdef CUDA
		, &props
#endif
		);

#ifdef CUDA
	int cpu = !strcmp(mode, "CPU");
	int gpu = !strcmp(mode, "GPU");
#else
	int cpu = 1;
	int gpu = 0;
#endif

	// Create test configuration.
	struct test_config_t* t = test_init(
		name, mode, n, nt, sx, sy, ss, rank, szcomm,
		xmin, ymin, zmin, xmax, ymax, zmax,
		bx, by, bs, ex, ey, es
#ifdef CUDA
		, &props
#endif
		);
	
	// Create another test configuration to check correctness.
	struct test_config_t* t_check = NULL;
#ifdef MPI
	if (t->rank == MPI_ROOT_NODE)
#endif
	{
		t_check = test_init(
			name, mode, n, nt, 1, 1, 1, 0, 1,
			xmin, ymin, zmin, xmax, ymax, zmax,
			bx, by, bs, ex, ey, es
#ifdef CUDA
			, &props
#endif
			);
	}
	
	// Generate the initial data disrtibution and load it
	// onto compute nodes.
	integer* array = (integer*)malloc(t->cpu.parent->grid->extsize * sizeof(integer));
	genirand(t->cpu.parent->grid->extsize, array);
	test_load(t, n, sx, sy, ss, sizeof(integer), (char*)array);
#ifdef MPI
	if (t->rank == MPI_ROOT_NODE)
#endif
	{
		size_t nxysb = n * n * n * sizeof(integer);
	
		// Copy the data array.
		memcpy(t_check->cpu.arrays[0], array, nxysb);
		
		// Duplicate initial distribution to the second level array.
		memcpy(t_check->cpu.arrays[1], t_check->cpu.arrays[0], nxysb);
	}
	free(array);
#ifdef VERBOSE
	printf("step 0\n");
	printf("step 1\n");
#endif
	// The time iterations loop, CPU and GPU versions.
	for (int it = 2; it < t->nt; it++)
	{
		// Run one iteration of the stencil, measuring its time.
		// In case of MPI, the time of iteration is measured together
		// with the time of data sync.
		struct timespec start, stop;
#ifdef MPI
		if (t->rank == MPI_ROOT_NODE)
#endif
		{
			stenfw_get_time(&start);
		}
#ifdef MPI
		struct grid_domain_t* subdomains = t->cpu.subdomains;

		int nsubdomains = t->cpu.nsubdomains;

		// Copy the current iteration data into boundary slices
		// and compute stencil in them.
		// Boundary slices themselves are subdomains with respect
		// to each MPI decomposition domains.
		{
			// Set subdomain data copying callbacks:
			// use simple memcpy in this case.
			for (int i = 0; i < nsubdomains; i++)
			{
				struct grid_domain_t* sub = subdomains + i;
				sub->scatter_memcpy = &grid_subcpy;
				sub->gather_memcpy = &grid_subcpy;
			}

			// Scatter domain edges for separate computation.
			grid_scatter(subdomains, &t->cpu, 0, LAYOUT_MODE_CUSTOM);
			
			// Process edges subdomains.
			for (int i = 0; i < nsubdomains; i++)
			{
				struct grid_domain_t* sub = subdomains + i;

				int nx = sub->grid[0].bx + sub->grid[0].nx + sub->grid[0].ex;
				int ny = sub->grid[0].by + sub->grid[0].ny + sub->grid[0].ey;
				int ns = sub->grid[0].bs + sub->grid[0].ns + sub->grid[0].es;

				isum13pt_cpu(nx, ny, ns,
					(integer(*)[ny][nx])sub->arrays[0],
					(integer(*)[ny][nx])sub->arrays[1],
					(integer(*)[ny][nx])sub->arrays[2]);
			}
		}
		
		// Start sharing boundary slices between linked subdomains.
		MPI_Request* reqs = (MPI_Request*)malloc(sizeof(MPI_Request) * 2 * nsubdomains);
		for (int i = 0; i < nsubdomains; i++)
		{
			struct grid_domain_t* subdomain = subdomains + i;
			struct grid_domain_t* neighbor = *(subdomain->links.dense[0]);

			assert(neighbor->grid[1].extsize == subdomain->grid[0].extsize);
		
			int szelem = sizeof(integer);

			size_t dnx = neighbor->grid[1].nx * szelem;			
			size_t dny = neighbor->grid[1].ny;
			size_t dns = neighbor->grid[1].ns;

			size_t snx = subdomain->grid[0].nx * szelem;
			size_t sbx = subdomain->grid[0].bx * szelem;
			size_t sex = subdomain->grid[0].ex * szelem;
			
			size_t sny = subdomain->grid[0].ny, sns = subdomain->grid[0].ns;
			size_t sby = subdomain->grid[0].by, sbs = subdomain->grid[0].bs;
			size_t sey = subdomain->grid[0].ey, ses = subdomain->grid[0].es;

			size_t soffset = sbx + (sbx + snx + sex) *
				(sby + sbs * (sby + sny + sey));

			struct grid_domain_t obuf;
			memset(&obuf, 0, sizeof(struct grid_domain_t));
			obuf.arrays = subdomain->arrays + 1;
			obuf.narrays = 1;
			obuf.offset = 0;
			obuf.grid[0].nx = dnx;
			obuf.grid[0].ny = dny;
			obuf.grid[0].ns = dns;
			obuf.grid->size = dnx * dny * dns;
		
			struct grid_domain_t scpy = *subdomain;
			scpy.arrays = subdomain->arrays + 2;
			scpy.narrays = 1;
			scpy.offset = soffset;
			scpy.grid[0].nx = sbx + snx + sex;
			scpy.grid[0].ny = sby + sny + sey;
			scpy.grid[0].ns = sbs + sns + ses;
			
			// Copy data to the temporary buffer.
			grid_subcpy(dnx, dny, dns, &obuf, &scpy);

			// Exchange temporary buffers with the subdomain neighbour.
			int subdomain_rank = grid_rank1d(subdomain->parent->parent, subdomain->parent->grid);
			int neighbor_rank = grid_rank1d(neighbor->parent->parent, neighbor->parent->grid);
			MPI_SAFE_CALL(MPI_Isend(subdomain->arrays[1], obuf.grid->size,
				MPI_BYTE, neighbor_rank, 0, MPI_COMM_WORLD, &reqs[2 * i]));
			MPI_SAFE_CALL(MPI_Irecv(subdomain->arrays[0], obuf.grid->size,
				MPI_BYTE, neighbor_rank, 0, MPI_COMM_WORLD, &reqs[2 * i + 1]));
#ifdef VERBOSE
			printf("sharing: send %d->%d\n", subdomain_rank, neighbor_rank);
			printf("sharing: recv %d->%d\n", neighbor_rank, subdomain_rank);
#endif
		}
#endif // MPI
		// Compute inner grid points of the subdomain.
		int nx = t->cpu.grid->bx + t->cpu.grid->nx + t->cpu.grid->ex;
		int ny = t->cpu.grid->by + t->cpu.grid->ny + t->cpu.grid->ey;
		int ns = t->cpu.grid->bs + t->cpu.grid->ns + t->cpu.grid->es;

		if (cpu)
		{
			isum13pt_cpu(nx, ny, ns,
				(integer(*)[ny][nx])t->cpu.arrays[0],
				(integer(*)[ny][nx])t->cpu.arrays[1],
				(integer(*)[ny][nx])t->cpu.arrays[2]);	
		}
#ifdef CUDA
		if (gpu)
		{
			isum13pt_gpu(nx, ny, ns,
				(integer*)t->gpu.arrays[0],
				(integer*)t->gpu.arrays[1],
				(integer*)t->gpu.arrays[2]);
#ifdef VISUALIZE
#ifndef CUDA_MAPPED
			// If GPU is not using mapped host memory, then need to fetch
			// the current iteration solution explicitly.
			// TODO: in case of MPI/CUDA/!MAPPED this copy must go AFTER
			// boundaries gathering.
			CUDA_SAFE_CALL(cudaMemcpy(t->cpu.arrays[2], t->gpu.arrays[2],
				t->gpu.grid->extsize * sizeof(real), cudaMemcpyDeviceToHost));
#endif // CUDA_MAPPED
#endif
		}
#endif // CUDA
#ifdef MPI
		// Wait for boundaries sharing completion.
		MPI_Status* statuses = (MPI_Status*)malloc(2 * nsubdomains * sizeof(MPI_Status));
		MPI_SAFE_CALL(MPI_Waitall(2 * nsubdomains, reqs, statuses));
		for (int i = 0; i < 2 * nsubdomains; i++)
			MPI_SAFE_CALL(statuses[i].MPI_ERROR);
		free(statuses);
		free(reqs);
		for (int i = 0; i < nsubdomains; i++)
		{
			struct grid_domain_t* subdomain = subdomains + i;
			
			int szelem = sizeof(integer);

			size_t dnx = subdomain->grid[1].nx * szelem;
			size_t dbx = subdomain->grid[1].bx * szelem;
			size_t dex = subdomain->grid[1].ex * szelem;
			
			size_t dny = subdomain->grid[1].ny, dns = subdomain->grid[1].ns;
			size_t dby = subdomain->grid[1].by, dbs = subdomain->grid[1].bs;
			size_t dey = subdomain->grid[1].ey, des = subdomain->grid[1].es;

			size_t doffset = dbx + (dbx + dnx + dex) *
				(dby + dbs * (dby + dny + dey));

			struct grid_domain_t dcpy = *subdomain;
			dcpy.arrays = subdomain->arrays + 2;
			dcpy.narrays = 1;
			dcpy.offset = doffset;
			dcpy.grid[0].nx = dbx + dnx + dex;
			dcpy.grid[0].ny = dby + dny + dey;
			dcpy.grid[0].ns = dbs + dns + des;

			struct grid_domain_t ibuf;
			memset(&ibuf, 0, sizeof(struct grid_domain_t));
			ibuf.arrays = subdomain->arrays;
			ibuf.narrays = 1;
			ibuf.offset = 0;
			ibuf.grid[0].nx = dnx;
			ibuf.grid[0].ny = dny;
			ibuf.grid[0].ns = dns;
		
			// Copy data to temporary buffer.
			grid_subcpy(dnx, dny, dns, &dcpy, &ibuf);

			// Swap pointers to make the last iteration in the bottom.
			char* w = subdomain->arrays[0];
			subdomain->arrays[0] = subdomain->arrays[2];
			subdomain->arrays[2] = w;
		}

		// Gather bounradies on for the next time step. Insert the
		// separately computed boundaries back into the sudomains
		// for the next time step.
		struct grid_domain_t target = t->cpu;
		target.narrays = 1;
		target.arrays = t->cpu.arrays + 2;
		grid_gather(&target, subdomains, 1, LAYOUT_MODE_CUSTOM);
		
		if (t->rank != MPI_ROOT_NODE)
		{
#ifdef VERBOSE
			printf("step %d\n", it);
#endif
		}
		else
#endif // MPI		
		{
			stenfw_get_time(&stop);
			printf("step %d time = ", it);
			stenfw_print_time_diff(start, stop);
			printf(" sec\n");
		}
#ifdef MPI
		if (t->rank == MPI_ROOT_NODE)
#endif
		{
			// Compute inner grid points of the control solution subdomain.
			int nx = t_check->cpu.grid->bx + t_check->cpu.grid->nx + t_check->cpu.grid->ex;
			int ny = t_check->cpu.grid->by + t_check->cpu.grid->ny + t_check->cpu.grid->ey;
			int ns = t_check->cpu.grid->bs + t_check->cpu.grid->ns + t_check->cpu.grid->es;

			isum13pt_cpu(nx, ny, ns,
				(integer(*)[ny][nx])t_check->cpu.arrays[0],
				(integer(*)[ny][nx])t_check->cpu.arrays[1],
				(integer(*)[ny][nx])t_check->cpu.arrays[2]);
		}

		// Print the stats of difference between the solution and
		// the control solution.
		test_write_imaxabsdiff(t, t_check, 2, it);

		// Swap pointers to rewrite the oldest iteration with
		// the next one.
		char* w = t->cpu.arrays[0];
		t->cpu.arrays[0] = t->cpu.arrays[1];
		t->cpu.arrays[1] = t->cpu.arrays[2];
		t->cpu.arrays[2] = w;
#ifdef CUDA
		if (gpu)
		{
			// Also swap the corresponding GPU arrays pointers.
			w = t->gpu.arrays[0];
			t->gpu.arrays[0] = t->gpu.arrays[1];
			t->gpu.arrays[1] = t->gpu.arrays[2];
			t->gpu.arrays[2] = w;
		}
#endif
#ifdef MPI
		if (t->rank == MPI_ROOT_NODE)
#endif
		{
			// Swap pointers to rewrite the oldest control solution
			// iteration with the next one.
			char* w = t_check->cpu.arrays[0];
			t_check->cpu.arrays[0] = t_check->cpu.arrays[1];
			t_check->cpu.arrays[1] = t_check->cpu.arrays[2];
			t_check->cpu.arrays[2] = w;
		}
	}

	// Dispose the test configurations.
#ifdef MPI
	if (t->rank == MPI_ROOT_NODE)
#endif
	{
		test_dispose(t_check);
	}
	test_dispose(t);

	return 0;
}

