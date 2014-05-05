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

#include "isum13pt_init.h"

#include <malloc.h>
#include <stdio.h>
#include <string.h>

#define ABS(a) ((a) > 0 ? (a) : -(a))

#ifdef VISUALIZE
#include <ImageMagick/magick/MagickCore.h>

#define SZCELL 10

static void draw_iabsdiff(int nx, int ny, int ns,
	integer* data1, integer* data2, int it)
{
	typedef struct
	{
		char red;
		char green;
		char blue;
	}
	RGB;

	// Create pixels buffer.
	int width = nx * SZCELL + 1, height = ny * SZCELL + 1;
	RGB* rgb = (RGB*)malloc(sizeof(RGB) * width * height);
	for (int j = 0; j < height; j++)
		for (int i = 0; i < width; i++)
		{
			rgb[i + j * width].red = 255;
			rgb[i + j * width].green = 255;
			rgb[i + j * width].blue = 255;
		}

	size_t np = nx * ny;
	for (int k = 0; k < ns; k++)
	{
		// Create image info template.
		ExceptionInfo exception;
		ImageInfo* iinfo = AcquireImageInfo();
	
		// Create background.
		MagickPixelPacket background;
		GetExceptionInfo(&exception);
		QueryMagickColor("white", &background, &exception);

		// Create new image.
		Image* image = NewMagickImage(iinfo, width, height, &background);

		// Set image filename.
		sprintf(image->filename, "%03d-%03d.png", it, k);

		for (int j = 0; j < ny; j++)
			for (int i = 0; i < nx; i++)
		{
			size_t idx = k * np + j * nx + i;		
			integer diff = ABS(data1[idx] - data2[idx]);
			if (diff)
			{
				for (int j1 = j * SZCELL + 1; j1 < (j + 1) * SZCELL - 1; j1++)
					for (int i1 = i * SZCELL + 1; i1 < (i + 1) * SZCELL - 1; i1++)
					{
						rgb[i1 + j1 * width].red = 255;
						rgb[i1 + j1 * width].green = 0;
						rgb[i1 + j1 * width].blue = 0;
					}
			}
			else
			{
				for (int j1 = j * SZCELL + 1; j1 < (j + 1) * SZCELL - 1; j1++)
					for (int i1 = i * SZCELL + 1; i1 < (i + 1) * SZCELL - 1; i1++)
					{
						rgb[i1 + j1 * width].red = 0;
						rgb[i1 + j1 * width].green = 255;
						rgb[i1 + j1 * width].blue = 0;
					}
			}
		}
		
		// Import buffer pixels to image.
		ImportImagePixels(image, 0, 0, width, height, "RGB", CharPixel, rgb);
		SyncAuthenticPixels(image, &exception);

		// Write image to file.
		WriteImage(iinfo, image);
	
		// Release used resources.
		DestroyImage(image);
		DestroyImageInfo(iinfo);
	}
	
	free(rgb);
}
#endif

void imaxabsdiff(int n, integer* data1, integer* data2,
	integer* diff, int* idiff, int* ndiff)
{
	*diff = ABS(data1[0] - data2[0]);
	*idiff = 0; *ndiff = 0;
	for (int i = 1; i < n; i++)
	{
		if (ABS(data1[i] - data2[i]) >= *diff)
		{
			*diff = ABS(data1[i] - data2[i]);
			*idiff = i;
		}
		if (ABS(data1[i] - data2[i]))
		{
			(*ndiff)++;
		}
	}
}

// Print the stats of difference between the solution and
// the control solution.
void test_write_imaxabsdiff(struct test_config_t* t,
	struct test_config_t* t_check, int iarray, int it)
{
	integer diff;
	int idiff, ndiff;
#ifdef CUDA
#ifndef CUDA_MAPPED
	if (!strcmp(t->mode, "GPU"))
	{
		CUDA_SAFE_CALL(cudaMemcpy(t->cpu.arrays[0], t->gpu.arrays[0],
			t->gpu.grid->extsize * sizeof(integer), cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaMemcpy(t->cpu.arrays[1], t->gpu.arrays[1],
			t->gpu.grid->extsize * sizeof(integer), cudaMemcpyDeviceToHost));
	}
#endif // CUDA_MAPPED
#endif // CUDA
#ifdef MPI
	// Send data to the root node for visualization.
	MPI_Request req;
	int src_rank = grid_rank1d(t->cpu.parent, t->cpu.grid);
	int dst_rank = 0;
	MPI_SAFE_CALL(MPI_Isend(t->cpu.arrays[iarray], t->cpu.grid->extsize * sizeof(integer), MPI_BYTE,
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
			domain->arrays = (char**)malloc(sizeof(char*) + domain->grid->extsize * sizeof(integer));
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
			MPI_SAFE_CALL(MPI_Irecv(domain->arrays[0], domain->grid->extsize * sizeof(integer), MPI_BYTE,
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
		integer* array = (integer*)malloc(t->cpu.parent->grid->extsize * sizeof(integer));

		// Gather current iteration data from the local subdomains.
		struct grid_domain_t target = *domains->parent;
		target.arrays = (char**)&array;
		target.narrays = 1;
		target.szelem = sizeof(integer);
		grid_gather(&target, domains, 0, LAYOUT_MODE_AUTO);
		
		// Dispose local root node buffers for domains data.
		for (int i = 0; i < ndomains; i++)
		{
			struct grid_domain_t* domain = domains + i;
			free(domain->arrays);
		}

		imaxabsdiff(t->cpu.parent->grid->extsize, array,
			(integer*)t_check->cpu.arrays[iarray], &diff, &idiff, &ndiff);
#ifdef VISUALIZE
		if (ndiff)
			draw_iabsdiff(t->cpu.parent->grid->nx,
				t->cpu.parent->grid->ny, t->cpu.parent->grid->ns, array,
				(integer*)t_check->cpu.arrays[iarray], it);
#endif
		free(array);
	}

	// Wait for asynchronous operations completion.
	MPI_Status status;
	status.MPI_ERROR = MPI_SUCCESS;
	MPI_SAFE_CALL(MPI_Wait(&req, &status));
	MPI_SAFE_CALL(status.MPI_ERROR);

	if (t->rank == MPI_ROOT_NODE)
#else
	imaxabsdiff(t->cpu.parent->grid->extsize, (integer*)t->cpu.arrays[iarray],
		(integer*)t_check->cpu.arrays[iarray], &diff, &idiff, &ndiff);
#ifdef VISUALIZE
	if (ndiff)		
		draw_iabsdiff(t->cpu.parent->grid->nx,
			t->cpu.parent->grid->ny, t->cpu.parent->grid->ns,
			(integer*)t->cpu.arrays[iarray],
			(integer*)t_check->cpu.arrays[iarray], it);
#endif
#endif // MPI
	{
		printf("%f%% different, max = %d @ %d\n",
			100.0 * ndiff / t->cpu.parent->grid->extsize, (int)diff, idiff);
	}
}

