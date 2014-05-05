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
#include <stdint.h>
#include <string.h>

// Create the simple decomposition grid.
struct grid_domain_t* grid_init_simple(
	int nx, int ny, int ns,	int sx, int sy, int ss,
	int bx, int ex, int by, int ey, int bs, int es)
{
	struct grid_domain_t* domains =
		grid_allocate(
			nx, ny, ns, sx, sy, ss,
			bx, ex, by, ey, bs, es);

	int nsubdomains = sx * sy * ss;

	grid_decompose_balanced(domains,
		nx, ny, ns, sx, sy, ss,
		bx, ex, by, ey, bs, es);

	// Set domains overlaps.
	grid_set_overlaps(domains, 0, sx, sy, ss,
		bx, ex, by, ey, bs, es);

	// Create an fictive empty domain
	// to be referenced in heighbourhood links.
	struct grid_domain_t* empty = NULL;
	size_t szempty = sizeof(struct grid_domain_t) * (PTRN_N + 1);
	empty = (struct grid_domain_t*)malloc(szempty);
	memset(empty, 0, szempty);
	empty->subdomains = (struct grid_domain_t*)(empty + 1);
	
	// Link domains using fictive empty domain.
	grid_set_links(domains, empty, 0, sx, sy, ss,
		bx, ex, by, ey, bs, es);

	// Set domains edges subgrid.
	grid_set_edges(domains, sx, sy, ss, 
		bx, ex, by, ey, bs, es);

	// Set domains links without fictive empty domain.
	grid_set_links(domains, 0, 0, sx, sy, ss,
		bx, ex, by, ey, bs, es);
	
	free(empty);

	// Multiply normalized overlaps by real.
	grid_overlaps_multiply(domains,
		nx, ny, ns, sx, sy, ss,
		bx, ex, by, ey, bs, es);

	// Calculate domains sizes.	
	grid_set_sizes(domains, nsubdomains + 1);
	
	// Perform consistency checks.
	grid_check_valid(domains, nsubdomains);
	
	return domains;
}

