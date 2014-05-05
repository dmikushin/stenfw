#include <malloc.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "grid.h"

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
	printf("Basic scatter/gather test\n\n");

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
	
	int ndomains = sx * sy * ss + 1;

	// Setup the grid and subdomains dimensions.
	struct grid_domain_t* domains =	grid_init_simple(
		nx, ny, ns, sx, sy, ss,
		bx, ex, by, ey, bs, es);
	
	for (int iv = 0; iv < ndomains; iv++)
		printf("domain %04d nx = %04d, ny = %04d, ns = %04d\n",	iv,
			domains[iv].grid[0].nx,	domains[iv].grid[0].ny,
			domains[iv].grid[0].ns);
	
	int nv = nx * ny * ns;

	// Create some random data.	
	int narrays = 10;
	long** arrays1 = setarrays(narrays, nv);
	long** arrays2 = setarrays(narrays, nv);
	
	// Allocate space for each domain data.
	for (int iv = 0; iv < ndomains; iv++)
	{
		domains[iv].arrays = (char**)malloc((sizeof(char*) +
			domains[iv].grid[0].extsize * sizeof(long)) * narrays);
		char* pdarrays = (char*)(domains[iv].arrays + narrays);
		for (int iarray = 0; iarray < narrays; iarray++)
		{
			domains[iv].arrays[iarray] = pdarrays;
			pdarrays += domains[iv].grid[0].extsize * sizeof(long);
		}
		domains[iv].scatter_memcpy = &grid_subcpy;
		domains[iv].gather_memcpy = &grid_subcpy;
	}

	// Scatter first array data to the subdomains.
	struct grid_domain_t target;
	target.arrays = (char**)arrays1;
	target.narrays = narrays;
	target.szelem = sizeof(long);
	grid_scatter(domains, &target, 0, LAYOUT_MODE_AUTO);

	// Gather subdomains data to the second array.
	target.arrays = (char**)arrays2;
	target.narrays = narrays;
	target.szelem = sizeof(long);
	grid_gather(&target, domains, 0, LAYOUT_MODE_AUTO);
	
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
				free(domains);
				return -1;
			}
		}
	}

	free(arrays1); free(arrays2);	
	free(domains);
	
	printf("Test passed.\n");
	
	return 0;
}

