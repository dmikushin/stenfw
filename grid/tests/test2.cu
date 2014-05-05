#include <assert.h>
#include <malloc.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C"
{
	#include "grid.h"
};

#ifdef INTEGER
#define dtype long
#endif

#ifdef SINGLE
#define dtype float
#endif

#ifdef DOUBLE
#define dtype double
#endif

#if (!defined(INTEGER) && !defined(SINGLE) && !defined(DOUBLE)) || \
	(defined(SINGLE) && defined(DOUBLE))
#error "The integer, single or double floating-precision mode must be set"
#endif

#define _C(val) ((val) + ix + nx * iy + np * iz)
#define _L(val) (_C(val) - 1)
#define _R(val) (_C(val) + 1)
#define _D(val) (_C(val) - nx)
#define _U(val) (_C(val) + nx)
#define _B(val) (_C(val) - np)
#define _F(val) (_C(val) + np)

#define CROSS(val) (*_C(val) + \
	*_L(val) + *_R(val) + \
	*_D(val) + *_U(val) + \
	*_B(val) + *_F(val));

// The number of GPUs used in test.
int ngpus;

// Apply 3d cross numerical pattern
// to the given data field (CUDA device kernel).
void __global__ cross3d_cuda_kernel(
	int nx, int ny, int nz,
	dtype* prev, dtype* curr /** OUT **/ )
{
	const int ix = threadIdx.x + 1;
	const int iy = blockIdx.x + 1;
	const int iz = blockIdx.y + 1;
	
	int np = nx * ny;

	*_C(curr) = CROSS(prev);
}

// Apply 3d cross numerical pattern
// to the given data field (CUDA device wrapper).
int cross3d_cuda(
	int nx, int ny, int nz,
	dtype* prev, dtype* curr /** OUT **/ )
{
	dim3 threadsInBlock(nx - 2, 1);
	dim3 blocksInGrid(ny - 2, nz - 2);
	
	cross3d_cuda_kernel<<<blocksInGrid, threadsInBlock>>>(
		nx, ny, nz, prev, curr);
	cudaThreadSynchronize();

	return 0;
}

// Apply 3d cross numerical pattern
// to the given data field (regular CPU version).
int cross3d(
	int nx, int ny, int nz,
	dtype* prev, dtype* curr /** OUT **/ )
{
	int np = nx * ny;

	for (int iz = 1; iz < nz - 1; iz++)
		for (int iy = 1; iy < ny - 1; iy++)
			for (int ix = 1; ix < nx - 1; ix++)
			{
				*_C(curr) = CROSS(prev);
			}

	return 0;
}

long** setarrays(int narrays, int szarray)
{
	long** arrays = (long**)malloc((sizeof(long*) +
		sizeof(long) * szarray) * narrays);
	long* parrays = (long*)(arrays + narrays);
	
	for (int iarray = 0; iarray < narrays; iarray++)
	{
		arrays[iarray] = parrays;
		for (int i = 0; i < szarray; i++)
			parrays[i] = i % 2;
		parrays += szarray;
	}
	return arrays;
}

void memcpy_cuda_load(int nx, int ny, int ns,
	struct grid_domain_t* dst, struct grid_domain_t* src)
{
	assert(dst); assert(src);

	int dnx = dst->grid[0].nx;
	int dny = dst->grid[0].ny;
	int dnp = dnx * dny;
	
	int snx = src->grid[0].nx;
	int sny = src->grid[0].ny;
	int snp = snx * sny;

	enum cudaMemcpyKind kind = cudaMemcpyHostToDevice;

	for (int iarray = 0; iarray < dst->narrays; iarray++)
		for (int is = 0; is < ns; is++)
			for (int iy = 0; iy < ny; iy++)
			{
				cudaError_t status = cudaMemcpy(
					dst->arrays[iarray] + dst->offset + is * dnp + iy * dnx,
					src->arrays[iarray] + src->offset + is * snp + iy * snx, nx, kind);
				assert(status == cudaSuccess);
			}
}

void memcpy_cuda_save(int nx, int ny, int ns,
	struct grid_domain_t* dst, struct grid_domain_t* src)
{
	assert(dst); assert(src);

	int dnx = dst->grid[0].nx;
	int dny = dst->grid[0].ny;
	int dnp = dnx * dny;
	
	int snx = src->grid[0].nx;
	int sny = src->grid[0].ny;
	int snp = snx * sny;

	enum cudaMemcpyKind kind = cudaMemcpyDeviceToHost;

	for (int iarray = 0; iarray < dst->narrays; iarray++)
		for (int is = 0; is < ns; is++)
			for (int iy = 0; iy < ny; iy++)
			{
				cudaError_t status = cudaMemcpy(
					dst->arrays[iarray] + dst->offset + is * dnp + iy * dnx,
					src->arrays[iarray] + src->offset + is * snp + iy * snx, nx, kind);
				assert(status == cudaSuccess);
			}
}

int main(int argc, char* argv[])
{
	printf("Hybrid scatter/gather test on cpu and gpu\n\n");
	
	if (argc != 7)
	{
		printf("Usage <nx> <ny> <ns> <sx> <sy> <ss>\n");
		return 0;
	}

	int nx = atoi(argv[1]);
	int ny = atoi(argv[2]);
	int ns = atoi(argv[3]);
	
	int sx = atoi(argv[4]);
	int sy = atoi(argv[5]);
	int ss = atoi(argv[6]);
	
	// The overhead boundary thickness, 1 is enough
	// for 7-point 3d cross.
	int bx = 1, by = 1, bs = 1;
	int ex = 1, ey = 1, es = 1;
	
	printf("Mapping problem grid %d x %d x %d onto %d x %d x %d compute grid\n",
		nx, ny, ns, sx, sy, ss);

	int ndomains = sx * sy * ss + 1;

#define MIN(a,b) ((a) > (b) ? (b) : (a))
	
	// Set gpus count (only one, without threads).
	ngpus = 0;
	cudaGetDeviceCount(&ngpus);
	if (ngpus)
	{
		ngpus = 1;
		cudaError_t status = cudaSetDevice(0);
		assert(status == cudaSuccess);
	}
	
	printf("Using %d GPU(s)\n", ngpus);
	
	// Setup the grid and subdomains dimensions.
	struct grid_domain_t* domains = grid_init_simple(
		nx, ny, ns, sx, sy, ss,
		bx, ex, by, ey, bs, es);
	
	for (int iv = 0; iv < ndomains; iv++)
		printf("domain %04d nx = %04d, ny = %04d, ns = %04d\n",
			iv, domains[iv].grid[0].nx, domains[iv].grid[0].ny, domains[iv].grid[0].ns);
	
	int nv = nx * ny * ns;

	// Allocate data arrays.
	int narrays = 2;
	long** arrays1 = setarrays(narrays, nv);
	long** arrays2 = setarrays(narrays, nv);

	// Allocate space for each domain data.
	ndomains--;
	for (int iv = 0; iv < ngpus; iv++)
	{	
		// Allocate arrays in devices memory.
		domains[iv].arrays = (char**)malloc(sizeof(char*) * narrays);
		for (int iarray = 0; iarray < narrays; iarray++)
		{
			cudaError_t status = cudaMalloc(
				(void**)&(domains[iv].arrays[iarray]),
				domains[iv].grid[0].extsize * sizeof(long));
			assert(status == cudaSuccess);
		}
		
		// Set data copying callbacks.
		domains[iv].scatter_memcpy = &memcpy_cuda_load;
		domains[iv].gather_memcpy = &memcpy_cuda_save;
	}

	for (int iv = ngpus; iv < ndomains; iv++)
	{
		domains[iv].arrays = (char**)malloc((sizeof(char*) +
			domains[iv].grid[0].extsize * sizeof(long)) * narrays);
		char* pdarrays = (char*)(domains[iv].arrays + narrays);
		for (int iarray = 0; iarray < narrays; iarray++)
		{
			domains[iv].arrays[iarray] = pdarrays;
			pdarrays += domains[iv].grid[0].extsize * sizeof(long);
		}
		
		// Set data copying callbacks.
		domains[iv].scatter_memcpy = &grid_subcpy;
		domains[iv].gather_memcpy = &grid_subcpy;
	}

	// Scatter first array data to the subdomains.
	struct grid_domain_t target;
	target.arrays = (char**)arrays1;
	target.narrays = narrays;
	target.szelem = sizeof(long);
	grid_scatter(domains, &target, 0, LAYOUT_MODE_AUTO);
	
	// Compute 3d cross pattern in global array (for results check)
	// using CPU implementation.
	cross3d(nx, ny, ns, arrays1[0], arrays1[1]);
	
	// Compute 3d cross pattern in subdomains.
	for (int iv = 0; iv < ngpus; iv++)
	{
		struct grid_domain_t* domain = domains + iv;
		
		cross3d_cuda(
			domain->grid[0].bx + domain->grid[0].nx + domain->grid[0].ex,
			domain->grid[0].by + domain->grid[0].ny + domain->grid[0].ey,
			domain->grid[0].bs + domain->grid[0].ns + domain->grid[0].es,
			(long*)(domain->arrays[0]), (long*)(domain->arrays[1]));
	}

	// Compute 3d cross pattern in subdomains.
	for (int iv = ngpus; iv < ndomains; iv++)
	{
		struct grid_domain_t* domain = domains + iv;
		
		cross3d(domain->grid[0].bx + domain->grid[0].nx + domain->grid[0].ex,
			domain->grid[0].by + domain->grid[0].ny + domain->grid[0].ey,
			domain->grid[0].bs + domain->grid[0].ns + domain->grid[0].es,
			(long*)(domain->arrays[0]), (long*)(domain->arrays[1]));
	}

	// Gather subdomains data to the second array.
	target.arrays = (char**)arrays2;
	target.narrays = narrays;
	target.szelem = sizeof(long);
	grid_gather(&target, domains, 0, LAYOUT_MODE_AUTO);
	
	// Check two groups of arrays are equal.
	const char* msg = "Array %d values difference trapped at %d: %ld != %ld\n";
	for (int iarray = 0; iarray < narrays; iarray++)
	{
		for (int i = 0; i < nv; i++)
		{
			if (arrays1[iarray][i] != arrays2[iarray][i])
			{
				printf(msg, iarray, i, 
					arrays1[iarray][i], arrays2[iarray][i]);
				free(arrays1); free(arrays2);
				for (int iv = 0; iv < ngpus; iv++)
				{
					for (int i = 0; i < narrays; i++)
						cudaFree(domains[iv].arrays[i]);
					free(domains[iv].arrays);
				}
				for (int iv = ngpus; iv < ndomains; iv++)
					free(domains[iv].arrays);
				free(domains);
				return -1;
			}
		}
	}

	free(arrays1); free(arrays2);
	for (int iv = 0; iv < ngpus; iv++)
	{
		for (int i = 0; i < narrays; i++)
			cudaFree(domains[iv].arrays[i]);
		free(domains[iv].arrays);
	}
	for (int iv = ngpus; iv < ndomains; iv++)
		free(domains[iv].arrays);
	free(domains);
	
	printf("Test passed.\n");
	
	return 0;
}

