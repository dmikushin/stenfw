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

void grid_decompose_balanced(
	struct grid_domain_t* domains,
	int nx, int ny, int ns, int sx, int sy, int ss,
	int bx, int ex, int by, int ey, int bs, int es)
{
	// Trap integer ranges.
	assert((nx > 0) && (ny > 0) && (ns > 0));
	assert((sx > 0) && (sy > 0) && (ss > 0));
	assert((bx >= 0) && (by >= 0) && (bs >= 0));
	assert((ex >= 0) && (ey >= 0) && (es >= 0));

	// Adjust grid if it is inappropriate for partitioning.
	if (sx > nx) sx = nx;
	if (sy > ny) sy = ny;
	if (ss > ns) ss = ns;	

	int nv = nx * ny * ns;
	int nsubdomains = sx * sy * ss;
	int sv = nsubdomains;

	// Calculate the subdomains dimensions.
	// TODO: probably add 16 or some other "good"
	// divide factor to favor aligned memory?
	int snx = nx / sx, srx = nx % sx;
	int sny = ny / sy, sry = ny % sy;
	int sns = ns / ss, srs = ns % ss;

	// Trap overhead size >= subdomain dimension.
	assert(bx + ex < snx);
	assert(by + ey < sny);
	assert(bs + es < sns);

	// Setup the subdomains dimensions.
	{
		int iv = 0;

		for (int is = 0; is < srs; is++)
		{
			for (int iy = 0; iy < sry; iy++)
			{
				for (int ix = 0; ix < srx; ix++, iv++)
				{
					domains[iv].grid[0].nx = snx + 1;
					domains[iv].grid[0].ny = sny + 1;
					domains[iv].grid[0].ns = sns + 1;
				}
				for (int ix = srx; ix < sx; ix++, iv++)
				{
					domains[iv].grid[0].nx = snx;
					domains[iv].grid[0].ny = sny + 1;
					domains[iv].grid[0].ns = sns + 1;
				}
			}
			for (int iy = sry; iy < sy; iy++)
			{
				for (int ix = 0; ix < srx; ix++, iv++)
				{
					domains[iv].grid[0].nx = snx + 1;
					domains[iv].grid[0].ny = sny;
					domains[iv].grid[0].ns = sns + 1;
				}
				for (int ix = srx; ix < sx; ix++, iv++)
				{
					domains[iv].grid[0].nx = snx;
					domains[iv].grid[0].ny = sny;
					domains[iv].grid[0].ns = sns + 1;
				}
			}
		}
		for (int is = srs; is < ss; is++)
		{
			for (int iy = 0; iy < sry; iy++)
			{
				for (int ix = 0; ix < srx; ix++, iv++)
				{
					domains[iv].grid[0].nx = snx + 1;
					domains[iv].grid[0].ny = sny + 1;
					domains[iv].grid[0].ns = sns;
				}
				for (int ix = srx; ix < sx; ix++, iv++)
				{
					domains[iv].grid[0].nx = snx;
					domains[iv].grid[0].ny = sny + 1;
					domains[iv].grid[0].ns = sns;
				}
			}
			for (int iy = sry; iy < sy; iy++)
			{
				for (int ix = 0; ix < srx; ix++, iv++)
				{
					domains[iv].grid[0].nx = snx + 1;
					domains[iv].grid[0].ny = sny;
					domains[iv].grid[0].ns = sns;
				}
				for (int ix = srx; ix < sx; ix++, iv++)
				{
					domains[iv].grid[0].nx = snx;
					domains[iv].grid[0].ny = sny;
					domains[iv].grid[0].ns = sns;
				}
			}
		}
		
		assert (iv == sv);
	}
	
	// Setup the subdomains grids offsets.
	for (int is = 0, os = 0, iv = 0; is < ss; is++, os += domains[iv].grid[0].ns)
		for (int iy = 0, oy = 0; iy < sy; iy++, oy += domains[iv].grid[0].ny)
			for (int ix = 0, ox = 0; ix < sx; ix++, ox += domains[iv].grid[0].nx, iv++)
			{
				domains[iv].grid[0].ox = ox;
				domains[iv].grid[0].oy = oy;
				domains[iv].grid[0].os = os;
			}
}

