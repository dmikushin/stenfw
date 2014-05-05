#include "grid.h"

#include <assert.h>
#include <string.h>

// Copy 3d source subdomain to destanation domain without
// buffering, using memcpy function.
void grid_subcpy(int nx, int ny, int ns,
	struct grid_domain_t* dst, struct grid_domain_t* src)
{
	assert(dst); assert(src);

	int dnx = dst->grid[0].nx;
	int dny = dst->grid[0].ny;
	int dnp = dnx * dny;
	
	int snx = src->grid[0].nx;
	int sny = src->grid[0].ny;
	int snp = snx * sny;

	for (int iarray = 0; iarray < dst->narrays; iarray++)
		for (int is = 0; is < ns; is++)
			for (int iy = 0; iy < ny; iy++)
				memcpy(dst->arrays[iarray] + dst->offset + is * dnp + iy * dnx,
					src->arrays[iarray] + src->offset + is * snp + iy * snx, nx);
}

