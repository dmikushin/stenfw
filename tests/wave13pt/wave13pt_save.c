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

#include "wave13pt_init.h"

#include <malloc.h>
#include <stdio.h>
#include <string.h>

#ifdef VISUALIZE

int vdfcreate(int argc, char* argv[]);

// Create a data file in UCAR Vapor format.
void test_create_vapor_vdf(struct test_config_t* t)
{
#ifdef MPI
	if (t->rank == MPI_ROOT_NODE)
#endif
	{
		// Prepare arguments for vdfcreate routine.
		int argc = 8;
		char* argv[] =
		{
			"vdfcreate",
			"-dimension", "%dx%dx%d",
			"-numts", "%d",
			"-vars3d", "w",
			"%s.vdf"
		};
	
		char dims[20];
		sprintf(dims, argv[2],
			t->cpu.parent->grid->nx,
			t->cpu.parent->grid->ny,
			t->cpu.parent->grid->ns);
		argv[2] = dims;
	
		char numts[10];
		sprintf(numts, argv[4], t->nt);
		argv[4] = numts;
	
		char* filename = (char*)malloc(strlen(t->name) + strlen(".vdf") + 1);
		sprintf(filename, argv[7], t->name);
		argv[7] = filename;
	
		// Create vdf file with UCAR Vapor vdfcreate.
		vdfcreate(argc, argv);
	
		free(filename);
	}
}

int raw2vdf(int argc, char* argv[]);

// Write step variables into the previously created
// UCAR Vapor format data file.
void test_write_vapor_vdf(struct test_config_t* t, int iarray, int itime)
{
#ifdef CUDA
#ifndef CUDA_MAPPED
	if (!strcmp(t->mode, "GPU"))
	{
		CUDA_SAFE_CALL(cudaMemcpy(t->cpu.arrays[0], t->gpu.arrays[0],
			t->gpu.grid->extsize * sizeof(real), cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaMemcpy(t->cpu.arrays[1], t->gpu.arrays[1],
			t->gpu.grid->extsize * sizeof(real), cudaMemcpyDeviceToHost));
	}
#endif // CUDA_MAPPED
#endif // CUDA
#ifdef MPI
	// Send data to the root node for visualization.
	MPI_Request req;
	int src_rank = grid_rank1d(t->cpu.parent, t->cpu.grid);
	int dst_rank = 0;
	MPI_SAFE_CALL(MPI_Isend(t->cpu.arrays[iarray], t->cpu.grid->extsize * sizeof(real), MPI_BYTE,
		dst_rank, 0, MPI_COMM_WORLD, &req));
#ifdef VERBOSE
	printf("save: send %d->%d\n", src_rank, dst_rank);
#endif
	if (t->rank == MPI_ROOT_NODE)
	{
		struct grid_domain_t* domains = t->domains;

		int ndomains = domains->parent->nsubdomains;
	
		// Create a local root node buffers for domains data.
		for (int i = 0; i < ndomains; i++)
		{
			struct grid_domain_t* domain = domains + i;
			domain->arrays = (char**)malloc(sizeof(char*) + domain->grid->extsize * sizeof(real));
			domain->arrays[0] = (char*)(domain->arrays + 1);
			domain->narrays = 1;
		}

		// Receive each subdomain data from its corresponding MPI node.
		MPI_Request* reqs = (MPI_Request*)malloc(ndomains * sizeof(MPI_Request));
		for (int i = 0; i < ndomains; i++)
		{
			struct grid_domain_t* domain = domains + i;
			int src_rank = grid_rank1d(domain->parent, domain->grid);
			int dst_rank = 0;
			MPI_SAFE_CALL(MPI_Irecv(domain->arrays[0], domain->grid->extsize * sizeof(real), MPI_BYTE,
				src_rank, 0, MPI_COMM_WORLD, &reqs[i]));
#ifdef VERBOSE
			printf("save: recv %d->%d\n", src_rank, dst_rank);
#endif
		}

		// Wait for asynchronous operations completion.
		MPI_Status* statuses = (MPI_Status*)malloc(ndomains * sizeof(MPI_Status));
		MPI_SAFE_CALL(MPI_Waitall(ndomains, reqs, statuses));
		for (int i = 0; i < ndomains; i++)
			MPI_SAFE_CALL(statuses[i].MPI_ERROR);
		free(statuses);
		free(reqs);
		
		// Create a temporary array to store the data for visualization.				
		real* array = (real*)malloc(t->cpu.parent->grid->extsize * sizeof(real));

		// Gather current iteration data from the local subdomains.
		struct grid_domain_t target = *domains->parent;
		target.arrays = (char**)&array;
		target.narrays = 1;
		target.szelem = sizeof(real);
		grid_gather(&target, domains, 0, LAYOUT_MODE_AUTO);
		
		// Dispose local root node buffers for domains data.
		for (int i = 0; i < ndomains; i++)
		{
			struct grid_domain_t* domain = domains + i;
			free(domain->arrays);
		}

		// Write variable.
		FILE* fp = fopen("data.bin", "wb");
		fwrite(array, 1, t->cpu.parent->grid->extsize * sizeof(real), fp);
		fclose(fp);

		free(array);
	}

	// Wait for asynchronous operations completion.
	MPI_Status status;
	status.MPI_ERROR = MPI_SUCCESS;
	MPI_SAFE_CALL(MPI_Wait(&req, &status));
	MPI_SAFE_CALL(status.MPI_ERROR);

	if (t->rank == MPI_ROOT_NODE)
#else
	// Write variable.
	FILE* fp = fopen("data.bin", "wb");
	fwrite(t->cpu.arrays[iarray], 1, t->cpu.parent->grid->extsize * sizeof(real), fp);
	fclose(fp);
	
#endif // MPI
	{
		int argc = 7;
		char* margv[] =
		{
			"raw2vdf",
			"-ts", "%d",
			"-varname", "",
			"%s.vdf", "data.bin"
		};
		char* argv[argc];

		char ts[10];
		sprintf(ts, margv[2], itime);
		margv[2] = ts;
		margv[4] = "w";
	
		char* filename = (char*)malloc(strlen(t->name) + strlen(".vdf") + 1);
		sprintf(filename, margv[5], t->name);
		margv[5] = filename;

		// Write to vdf file with UCAR Vapor raw2vdf.
		memcpy(argv, margv, argc * sizeof(char*));
		raw2vdf(argc, argv);
	
		free(filename);
	}
}

#endif // VISUALIZE

