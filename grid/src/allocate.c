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
#include <malloc.h>
#include <string.h>

// Allocate space and perform initial setup of domains
// configuration. Memory allocation includes space for
// nested subdomains and links: parent domain + 
// nsubdomains + PTRN_N nested domains in each subdomain,
struct grid_domain_t* grid_allocate(
	int nx, int ny, int ns,	int sx, int sy, int ss,
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

	int nsubdomains = sx * sy * ss;

	size_t szdomains = sizeof(struct grid_domain_t) *
		((PTRN_N + 1) * nsubdomains + 1);
	struct grid_domain_t* domains =
		(struct grid_domain_t*)malloc(szdomains);

	// Zero domains config space.
	memset(domains, 0, szdomains);

	// Put the parent domain on the last place.
	struct grid_domain_t* parent = domains + nsubdomains;

	// Record global grid into parent domain.
	struct grid_t* grid = parent->grid;
	grid->nx = nx; grid->ny = ny; grid->ns = ns;	
	parent->grid[1] = parent->grid[0];
	
	parent->parent = NULL;
	parent->subdomains = domains;

	parent->sx = sx;
	parent->sy = sy;
	parent->ss = ss;
	parent->nsubdomains = nsubdomains;

	// Get space for subdomains from the previously
	// allocated memory.
	struct grid_domain_t* subdomains = domains + nsubdomains + 1;
	
	// Set pointers for subdomains.
	for (int is = 0, iv = 0; is < ss; is++)
		for (int iy = 0; iy < sy; iy++)
			for (int ix = 0; ix < sx; ix++, iv++)
			{
				struct grid_domain_t* domain = domains + iv;
				domain->subdomains = subdomains + iv * PTRN_N;
				domain->parent = parent;
				for (int i = 0; i < PTRN_N; i++)
					domain->subdomains[i].parent = domain;
				
				domain->grid[0].ix = ix;
				domain->grid[0].iy = iy;
				domain->grid[0].is = is;
			}

	return domains;
}

