/*
 * COACCEL - the generalized framework for distributed computing on accelerators
 *
 * Copyright (c) 2010 Dmitry Mikushin
 *
 * This software is provided 'as-is', without any express or implied warranty.
 * In no event will the authors be held liable for any damages arising 
 * from the use of this software.
 * Permission is granted to anyone to use this software for any purpose, 
 * including commercial applications, and to alter it and redistribute it freely, 
 * subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented;
 * you must not claim that you wrote the original software.
 * If you use this software in a product, an acknowledgment
 * in the product documentation would be appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such,
 * and must not be misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 */

#include "grid.h"

#include <assert.h>

// Multiply the grid domains overlaps by the specified values.
void grid_overlaps_multiply(
	struct grid_domain_t* domains,
	int nx, int ny, int ns, int sx, int sy, int ss,
	int bx, int ex, int by, int ey, int bs, int es)
{
	// XXX: Trap overhead size >= dimension,
	// it is not possible to handle this case correctly.
	assert((sx == 1) || ((sx > 1) && (bx < nx) && (ex < nx)));
	assert((sy == 1) || ((sy > 1) && (by < ny) && (ey < ny)));
	assert((ss == 1) || ((ss > 1) && (bs < ns) && (es < ns)));

	// Adjust grid if it is inappropriate for partitioning.
	if (sx > nx) sx = nx;
	if (sy > ny) sy = ny;
	if (ss > ns) ss = ns;

	// XXX: Handle a special case:
	// when domains grid dimension = 1
	// we cannot put any overhead, other than zero.
	int incx = 1, incy = 1, incs = 1;
	if (sx == 1) { bx = 0; ex = 0; incx = 0; }
	if (sy == 1) { by = 0; ey = 0; incy = 1 - sx; }
	if (ss == 1) { bs = 0; es = 0; incs = 1 - sx * sy; }

	int sv = sx * sy * ss;

	// Multiply domains overheads by the actual
	// overheads values.
	for (int iv = 0; iv < sv; iv++)
	{
		struct grid_domain_t* domain = domains + iv;

		domain->grid[0].bx *= bx; domain->grid[0].ex *= ex;
		domain->grid[0].by *= by; domain->grid[0].ey *= ey;
		domain->grid[0].bs *= bs; domain->grid[0].es *= es;

		domain->grid[1] = domain->grid[0];

		int nsubdomains = domain->nsubdomains;
		for (int i = 0; i < nsubdomains; i++)
		{
			struct grid_domain_t* sub = domain->subdomains + i;
		
			sub->grid[0].bx *= bx; sub->grid[0].ex *= ex;
			sub->grid[0].by *= by; sub->grid[0].ey *= ey;
			sub->grid[0].bs *= bs; sub->grid[0].es *= es;			

			sub->grid[1].bx *= bx; sub->grid[1].ex *= ex;
			sub->grid[1].by *= by; sub->grid[1].ey *= ey;
			sub->grid[1].bs *= bs; sub->grid[1].es *= es;
		}
	}
}

