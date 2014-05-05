#include <assert.h>
#include <malloc.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <stdint.h>

#include "grid.h"

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
#define _L(val) ((val) - 1)
#define _R(val) ((val) + 1)
#define _D(val) ((val) - nx)
#define _U(val) ((val) + nx)
#define _B(val) ((val) - np)
#define _F(val) ((val) + np)

// Apply 3d cross numerical pattern
// to the given data field (regular CPU version).
int cross3d(
	int nx, int ny, int nz,
	int bx, int by, int bs,
	int ex, int ey, int es,
	dtype* prev, dtype* curr /** OUT **/ )
{
	int np = nx * ny;

	for (int iz = bs; iz < nz - es; iz++)
		for (int iy = by; iy < ny - ey; iy++)
			for (int ix = bx; ix < nx - ex; ix++)
			{
				dtype val = *_C(prev), *ptr;
				
				// Account pattern left extent values.
				ptr = _C(prev);
				for (int i = 0; i < bx; i++)
				{
					ptr = _L(ptr);
					val += *ptr;
				}
				
				// Account pattern right extent values.
				ptr = _C(prev);
				for (int i = 0; i < ex; i++)
				{
					ptr = _R(ptr);
					val += *ptr;
				}

				// Account pattern lower extent values.
				ptr = _C(prev);
				for (int i = 0; i < by; i++)
				{
					ptr = _D(ptr);
					val += *ptr;
				}
				
				// Account pattern upper extent values.
				ptr = _C(prev);
				for (int i = 0; i < ey; i++)
				{
					ptr = _U(ptr);
					val += *ptr;
				}

				// Account pattern back extent values.
				ptr = _C(prev);
				for (int i = 0; i < bs; i++)
				{
					ptr = _B(ptr);
					val += *ptr;
				}
				
				// Account pattern front extent values.
				ptr = _C(prev);
				for (int i = 0; i < es; i++)
				{
					ptr = _F(ptr);
					val += *ptr;
				}	
				
				*_C(curr) = val;
			}

	return 0;
}

long** setarrays(int narrays, int szarray)
{
	long** arrays = (long**)malloc((sizeof(long*) +
		sizeof(long) * szarray) * narrays);
	long* parrays = (long*)(arrays + narrays);
	double drandmax = (double)RAND_MAX;
	for (int i = 0; i < szarray * narrays; i++)
		parrays[i] = (long)round(rand() / drandmax);
	for (int iarray = 0; iarray < narrays; iarray++)
	{
		arrays[iarray] = parrays;
		parrays += szarray;
	}
	return arrays;
}

int main(int argc, char* argv[])
{
	printf("Cross3d integration with only initial scatter/gather\n");
	printf("and subdomains local boundaries sync on each step\n\n");

	if (argc != 13)
	{
		printf("Usage <nx> <ny> <ns> <sx> <sy> <ss>\n");
		printf("<bx> <ex> <by> <ey> <bs> <es>\n");
		return 0;
	}

	int nx = atoi(argv[1]);
	int ny = atoi(argv[2]);
	int ns = atoi(argv[3]);
	
	int sx = atoi(argv[4]);
	int sy = atoi(argv[5]);
	int ss = atoi(argv[6]);
	
	int bx = atoi(argv[7]), ex = atoi(argv[8]);
	int by = atoi(argv[9]), ey = atoi(argv[10]);
	int bs = atoi(argv[11]), es = atoi(argv[12]);
	
	printf("Mapping problem grid %d x %d x %d onto %d x %d x %d compute grid\n",
		nx, ny, ns, sx, sy, ss);
	printf("Using [%d, %d] x [%d, %d] x [%d, %d] fictive boundaries\n",
		bx, ex, by, ey, bs, es);

	size_t szelem = sizeof(dtype);

	int ndomains = sx * sy * ss + 1;
	
	// Setup the grid and subdomains dimensions.
	struct grid_domain_t* domains = grid_init_simple(
		nx, ny, ns, sx, sy, ss,
		bx, ex, by, ey, bs, es);
	
	for (int iv = 0; iv < ndomains; iv++)
		printf("domain %04d nx = %04d, ny = %04d, ns = %04d\n",	iv,
			domains[iv].grid[0].nx,	domains[iv].grid[0].ny,
			domains[iv].grid[0].ns);
	
	int nv = nx * ny * ns;

	// Allocate data arrays.
	int narrays = 2;
	long** arrays1 = setarrays(narrays, nv);
	long** arrays2 = setarrays(narrays, nv);

	// Allocate space for each domain data.
	ndomains--;
	for (int iv = 0; iv < ndomains; iv++)
	{
		struct grid_domain_t* domain = domains + iv;
		struct grid_domain_t* subdomains = domain->subdomains;

		int nsubdomains = domain->nsubdomains;

		// Count and allocate the required memory for each domain.
		size_t szarrays = narrays * (sizeof(char*) +
			domain->grid[0].extsize * szelem);
		for (int i = 0; i < nsubdomains; i++)
			szarrays += narrays * (sizeof(char*) +
				subdomains[i].grid[0].extsize * szelem);
		domain->arrays = (char**)malloc(szarrays);
		
		char* pdarrays = (char*)(domain->arrays + narrays);
		for (int iarray = 0; iarray < narrays; iarray++)
		{
			domain->arrays[iarray] = pdarrays;
			pdarrays += domain->grid[0].extsize * szelem;
		}
		for (int i = 0; i < nsubdomains; i++)
		{
			struct grid_domain_t* subdomain = subdomains + i;
		
			subdomain->arrays = (char**)pdarrays;
			pdarrays = (char*)(subdomain->arrays + narrays);
			for (int iarray = 0; iarray < narrays; iarray++)
			{
				subdomain->arrays[iarray] = pdarrays;
				pdarrays += subdomain->grid[0].extsize * szelem;
			}

			// Set subdomain data copying callbacks.
			subdomain->scatter_memcpy = &grid_subcpy;
			subdomain->gather_memcpy = &grid_subcpy;
		}
		
		// Set domain data copying callbacks.
		domain->scatter_memcpy = &grid_subcpy;
		domain->gather_memcpy = &grid_subcpy;
	}

	// Scatter first array data to the subdomains.
	struct grid_domain_t target;
	target.arrays = (char**)arrays1;
	target.narrays = narrays;
	target.szelem = sizeof(long);
	grid_scatter(domains, &target, 0, LAYOUT_MODE_AUTO);
	
	int nt = 10;
	
	for (int it = 0; it < nt; it++)
	{
		printf("step %03d\n", it);
	
		// Compute 3d cross pattern in global array (for results check)
		// using CPU implementation.
		cross3d(nx, ny, ns, bx, by, bs, ex, ey, es,
			arrays1[0], arrays1[1]);
		
		// Swap global arrays.
		long* swap = arrays1[0];
		arrays1[0] = arrays1[1];
		arrays1[1] = swap;

		// Compute 3d cross pattern in subdomains' edges.
		for (int iv = 0; iv < ndomains; iv++)
		{
			struct grid_domain_t* domain = domains + iv;
			struct grid_domain_t* subdomains = domain->subdomains;
		
			int nsubdomains = domain->nsubdomains;

			// Scatter domain edges for separate computation.
			target.arrays = domain->arrays;
			target.narrays = narrays;
			target.szelem = sizeof(long);
			grid_scatter(subdomains, &target, 0, LAYOUT_MODE_CUSTOM);

			// Process edges subdomains.
			for (int i = 0; i < nsubdomains; i++)
			{
				struct grid_domain_t* sub = subdomains + i;
				cross3d(sub->grid[0].bx + sub->grid[0].nx + sub->grid[0].ex,
					sub->grid[0].by + sub->grid[0].ny + sub->grid[0].ey,
					sub->grid[0].bs + sub->grid[0].ns + sub->grid[0].es,
					bx, by, bs, ex, ey, es,
					(long*)(sub->arrays[0]), (long*)(sub->arrays[1]));
			}
		}

		// Share edges between linked subdomains.
		for (int iv = 0; iv < ndomains; iv++)
		{
			struct grid_domain_t* domain = domains + iv;
			struct grid_domain_t* subdomains = domain->subdomains;

			int nsubdomains = domain->nsubdomains;

			for (int i = 0; i < nsubdomains; i++)
			{
				struct grid_domain_t* src = subdomains + i;
				struct grid_domain_t* dst = *(src->links.dense[0]);

				assert(dst->grid[1].extsize == src->grid[0].extsize);

				size_t dnx = dst->grid[1].nx * szelem;
				size_t dbx = dst->grid[1].bx * szelem;
				size_t dex = dst->grid[1].ex * szelem;
				
				size_t dny = dst->grid[1].ny, dns = dst->grid[1].ns;
				size_t dby = dst->grid[1].by, dbs = dst->grid[1].bs;
				size_t dey = dst->grid[1].ey, des = dst->grid[1].es;

				size_t doffset = dbx + (dbx + dnx + dex) *
					(dby + dbs * (dby + dny + dey));

				size_t snx = src->grid[0].nx * szelem;
				size_t sbx = src->grid[0].bx * szelem;
				size_t sex = src->grid[0].ex * szelem;
				
				size_t sny = src->grid[0].ny, sns = src->grid[0].ns;
				size_t sby = src->grid[0].by, sbs = src->grid[0].bs;
				size_t sey = src->grid[0].ey, ses = src->grid[0].es;

				size_t soffset = sbx + (sbx + snx + sex) *
					(sby + sbs * (sby + sny + sey));

				struct grid_domain_t dcpy = *dst;
				dcpy.arrays = dst->arrays;
				dcpy.narrays = 1;
				dcpy.offset = doffset;
				dcpy.grid[0].nx = dbx + dnx + dex;
				dcpy.grid[0].ny = dby + dny + dey;
				dcpy.grid[0].ns = dbs + dns + des;
			
				struct grid_domain_t scpy = *src;
				scpy.arrays = src->arrays + 1;
				scpy.narrays = 1;
				scpy.offset = soffset;
				scpy.grid[0].nx = sbx + snx + sex;
				scpy.grid[0].ny = sby + sny + sey;
				scpy.grid[0].ns = sbs + sns + ses;

				grid_subcpy(dnx, dny, dns, &dcpy, &scpy);
			}
		}

		// Compute 3d cross pattern in domains.
		for (int iv = 0; iv < ndomains; iv++)
		{
			struct grid_domain_t* domain = domains + iv;
			struct grid_domain_t* subdomains = domain->subdomains;

			cross3d(domain->grid[0].bx + domain->grid[0].nx + domain->grid[0].ex,
				domain->grid[0].by + domain->grid[0].ny + domain->grid[0].ey,
				domain->grid[0].bs + domain->grid[0].ns + domain->grid[0].es,
				bx, by, bs, ex, ey, es,
				(long*)(domain->arrays[0]), (long*)(domain->arrays[1]));

			// Swap domain arrays.
			char* swap = domain->arrays[0];
			domain->arrays[0] = domain->arrays[1];
			domain->arrays[1] = swap;

			// Sync domain 0 level arrays with the separately
			// computed edges, using secondary subdomains grid.
			target.arrays = domain->arrays;
			target.narrays = 1;
			target.szelem = sizeof(long);
			grid_gather(&target, subdomains, 1, LAYOUT_MODE_CUSTOM);
		}

	}

	// Gather subdomains data to the second array.
	target.arrays = (char**)arrays2;
	target.narrays = narrays;
	target.szelem = sizeof(long);
	grid_gather(&target, domains, 0, LAYOUT_MODE_AUTO);
	
	// Check two resulting (index level = 0) arrays are equal.
	const char* msg = "Array %d values difference trapped at %d: %ld != %ld\n";
	for (int iarray = 0, i = 0; i < nv; i++)
	{
		if (arrays1[iarray][i] != arrays2[iarray][i])
		{
			printf(msg, iarray, i, 
				arrays1[iarray][i], arrays2[iarray][i]);
			for (int k = 0; k < ns; k++)
			{
				for (int j = 0; j < ny; j++)
				{
					for (int i = 0; i < nx; i++)
					{
						int index = i + nx * j + k * nx * ny;
						printf("%04ld ", 
							arrays1[iarray][index] -
							arrays2[iarray][index]);
					}
					printf("\n");
				}
				printf("\n");
			}
			free(arrays1); free(arrays2);
			for (int iv = 0; iv < ndomains; iv++)
				free(domains[iv].arrays);
			free(domains);
			return -1;
		}
	}

	free(arrays1); free(arrays2);
	for (int iv = 0; iv < ndomains; iv++)
		free(domains[iv].arrays);
	free(domains);

	printf("Test passed.\n");
	
	return 0;
}

