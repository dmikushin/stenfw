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

#include <malloc.h>
#include <stdio.h>
#include <string.h>

// Distribute the specified initial data across MPI nodes.
// Method: first create the whole dataset on the root node,
// and then distribute it across compute nodes. Useful when
// bridging with serial application.
void test_load(struct test_config_t* t, int n, int sx, int sy, int ss,
	int szelem, char* data)
{
#ifndef MPI
	size_t nxysb = n * n * n * szelem;
	
	// Copy the data array.
	memcpy(t->cpu.arrays[0], data, nxysb);
#else
	struct grid_domain_t* domains = t->domains;
	
	int ndomains = domains->parent->nsubdomains;

	MPI_Request* reqs = NULL;
	if (t->rank == MPI_ROOT_NODE)
	{
		printf("Mapping problem grid %d x %d x %d onto %d x %d x %d compute grid\n",
			n, n, n, sx, sy, ss);
		
		// Create a local root node buffers for domains data.
		for (int i = 0; i < ndomains; i++)
		{
			struct grid_domain_t* domain = domains + i;
			domain->arrays = (char**)malloc(sizeof(char*) + domain->grid->extsize * szelem);
			domain->arrays[0] = (char*)(domain->arrays + 1);
			domain->narrays = 1;
		}

		// Scatter initial distribution to the local subdomains.
		struct grid_domain_t source = *domains->parent;
		source.arrays = &data;
		source.narrays = 1;
		source.szelem = szelem;
		grid_scatter(domains, &source, 0, LAYOUT_MODE_AUTO);
		
		// Send out each subdomain data to its corresponding MPI node.
		reqs = (MPI_Request*)malloc(ndomains * sizeof(MPI_Request));
		for (int i = 0; i < ndomains; i++)
		{
			struct grid_domain_t* domain = domains + i;
			int src_rank = 0;
			int dst_rank = grid_rank1d(domain->parent, domain->grid);
			MPI_SAFE_CALL(MPI_Isend(domain->arrays[0], domain->grid->extsize * szelem, MPI_BYTE,
				dst_rank, 0, MPI_COMM_WORLD, &reqs[i]));
#ifdef VERBOSE
			printf("load: send %d->%d\n", src_rank, dst_rank);
#endif
		}
	}

	MPI_Request req;
	int src_rank = 0;
	int dst_rank = grid_rank1d(t->cpu.parent, t->cpu.grid);
	MPI_SAFE_CALL(MPI_Irecv(t->cpu.arrays[0], t->cpu.grid->extsize * szelem, MPI_BYTE,
		src_rank, 0, MPI_COMM_WORLD, &req));
#ifdef VERBOSE
	printf("load: recv %d->%d\n", src_rank, dst_rank);
#endif
	// Wait for asynchronous operations completion.
	MPI_Status status;
	status.MPI_ERROR = MPI_SUCCESS;
	MPI_SAFE_CALL(MPI_Wait(&req, &status));
	MPI_SAFE_CALL(status.MPI_ERROR);
	if (t->rank == MPI_ROOT_NODE)
	{
		MPI_Status* statuses = (MPI_Status*)malloc(ndomains * sizeof(MPI_Status));
		MPI_SAFE_CALL(MPI_Waitall(ndomains, reqs, statuses));
		for (int i = 0; i < ndomains; i++)
			MPI_SAFE_CALL(statuses[i].MPI_ERROR);
		free(statuses);
		free(reqs);

		// Dispose local root node buffers for domains data.
		for (int i = 0; i < ndomains; i++)
		{
			struct grid_domain_t* domain = domains + i;
			free(domain->arrays);
		}
		
		// Restore original entire node domain.
		//memcpy(domains + t->rank, &t.cpu, sizeof(struct grid_domain_t));
	}

	size_t nxysb = t->cpu.grid->extsize * szelem;
#endif // MPI
	
	// Duplicate initial distribution to the second level array.
	memcpy(t->cpu.arrays[1], t->cpu.arrays[0], nxysb);
#if defined(CUDA)
#if !defined(CUDA_MAPPED)
	if (!strcmp(t->mode, "GPU"))
	{
		CUDA_SAFE_CALL(cudaMemcpy(t->gpu.arrays[0], t->cpu.arrays[0], nxysb, cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(t->gpu.arrays[1], t->cpu.arrays[1], nxysb, cudaMemcpyHostToDevice));
	}
#endif // CUDA_MAPPED
#endif // CUDA
}

