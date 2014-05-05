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
#include <stdint.h>
#include <string.h>

// Set domains edges subgrid.
void grid_set_edges(
	struct grid_domain_t* domains,
	int sx, int sy, int ss,
	int bx, int ex, int by, int ey, int bs, int es)
{
	int nsubdomains = sx * sy * ss;

	// XXX: Handle a special case:
	// when domains grid dimension = 1
	// we cannot put any overhead, other than zero.
	int incx = 1, incy = 1, incs = 1;
	if (sx == 1) { bx = 0; ex = 0; incx = 0; }
	if (sy == 1) { by = 0; ey = 0; incy = 1 - sx; }
	if (ss == 1) { bs = 0; es = 0; incs = 1 - sx * sy; }

	// Normalized versions of fictive boundaries
	// thincknesses.
	int nbx = bx ? 1 : 0, nby = by ? 1 : 0, nbs = bs ? 1 : 0;
	int nex = ex ? 1 : 0, ney = ey ? 1 : 0, nes = es ? 1 : 0;

	// Create local subgrids for each first level
	// grid domain.	
	for (int iv = 0; iv < nsubdomains; iv++)
	{
		struct grid_domain_t* domain = domains + iv;
		
		int nx = domain->grid[0].nx;
		int ny = domain->grid[0].ny;
		int ns = domain->grid[0].ns;
		
		nx += domain->grid[0].bx * bx + domain->grid[0].ex * ex;
		ny += domain->grid[0].by * by + domain->grid[0].ey * ey;
		ns += domain->grid[0].bs * bs + domain->grid[0].es * es;

		// Calculate domain nested grid dimensions.
		domain->sx = domain->grid[0].bx + domain->grid[0].ex;
		domain->sy = domain->grid[0].by + domain->grid[0].ey;
		domain->ss = domain->grid[0].bs + domain->grid[0].es;
		
		// Create subgrid of inner boundary only domains,
		// specialized for inter-domain boundaries exchanging:
		// each domain must have b*/e* fictive border thickness
		// both from the parent edge and from the inner side.
		domain->nsubdomains = 0;

		// Copy the configured grid structure to the
		// subdomain, and set sparse & dense pattern
		// links for it.
		#define LINK(ptrn, mask, offset) \
		{ \
			domain->subdomains[idomain].grid[0] = grids[0]; \
			domain->subdomains[idomain].grid[1] = grids[1]; \
		\
			domain->subdomains[idomain].parent = domain; \
			domain->subdomains[idomain].links.sparse[ptrn] = \
				(struct grid_domain_t*)(mask * (intptr_t)( \
				domain->links.sparse[ptrn]->subdomains + offset)); \
		\
			int* ndense = &(domain->subdomains[idomain].links.ndense); \
			assert(*ndense < PTRN_N); \
			domain->subdomains[idomain].links.dense[*ndense] = \
				(struct grid_domain_t**)(mask * (intptr_t)( \
				&(domain->subdomains[idomain].links.sparse[ptrn]))); \
			*ndense += mask; idomain += mask; \
			domain->nsubdomains += mask; \
		}

		// Set the grid configuration.
		#define GRID(i, \
			_bx, _nx, _ex, _by, _ny, _ey, \
			_bs, _ns, _es, _ix, _iy, _is) \
		{ \
			struct grid_t* grid = grids + i; \
			memset(grid, 0, sizeof(struct grid_t)); \
			grid->bx = _bx; grid->nx = _nx; grid->ex = _ex; \
			grid->by = _by; grid->ny = _ny; grid->ey = _ey; \
			grid->bs = _bs; grid->ns = _ns; grid->es = _es; \
			grid->ix = _ix; grid->iy = _iy; grid->is = _is; \
		}

		// Setup subdomains. Not all subdomains exist
		// for boundary domains, they are simply rewritten,
		// if the corresponding flag is zero.
		{
			int idomain = 0;
			
			// For each subdomain prepare both primary and
			// secondary grids. Secondary grid is adjusted
			// to accept the side boundary in neighbours'
			// edges synchronization process.
			struct grid_t grids[2];

			// Edge (0, Y, Z) subdomain.			
			// Link LEFT edge to the RIGHT edge
			// of the LEFT neighbour.
			GRID(0, 1, bx, 1, 0, ny, 0, 0, ns, 0, bx, 0, 0);
			GRID(1, 0, bx, 2, 0, ny, 0, 0, ns, 0, 0, 0, 0);
			LINK(PTRN_L, domain->grid[0].bx * nbx,
				domain->links.sparse[PTRN_L]->grid[0].bx);

			// Edge (1, Y, Z) subdomain.
			// Link RIGHT edge to the LEFT edge
			// of the RIGHT neighbour.
			GRID(0, 1, ex, 1, 0, ny, 0, 0, ns, 0, nx - 2 * ex, 0, 0);
			GRID(1, 2, ex, 0, 0, ny, 0, 0, ns, 0, nx - ex, 0, 0);
			LINK(PTRN_R, domain->grid[0].ex * nex, 0);

			// Edge (X, 0, Z) subdomain.
			// Link LOWER edge to the UPPER edge
			// of the LOWER neighbour.
			GRID(0, 0, nx, 0, 1, by, 1, 0, ns, 0, 0, by, 0);
			GRID(1, 0, nx, 0, 0, by, 2, 0, ns, 0, 0, 0, 0);
			LINK(PTRN_D, domain->grid[0].by * nby,
				domain->links.sparse[PTRN_D]->grid[0].bx +
				domain->links.sparse[PTRN_D]->grid[0].ex +
				domain->links.sparse[PTRN_D]->grid[0].by);

			// Edge (X, 1, Z) subdomain.
			// Link UPPER edge to the LOWER edge
			// of the UPPER neighbour.
			GRID(0, 0, nx, 0, 1, ey, 1, 0, ns, 0, 0, ny - 2 * ey, 0);
			GRID(1, 0, nx, 0, 2, ey, 0, 0, ns, 0, 0, ny - ey, 0);
			LINK(PTRN_U, domain->grid[0].ey * ney,
				domain->links.sparse[PTRN_U]->grid[0].bx +
				domain->links.sparse[PTRN_U]->grid[0].ex);

			// Edge (X, Y, 0) subdomain.
			// Link BACK edge to the FRONT edge
			// of the BACK neighbour.
			GRID(0, 0, nx, 0, 0, ny, 0, 1, bs, 1, 0, 0, bs);
			GRID(1, 0, nx, 0, 0, ny, 0, 0, bs, 2, 0, 0, 0);
			LINK(PTRN_B, domain->grid[0].bs * nbs,
				domain->links.sparse[PTRN_B]->grid[0].bx +
				domain->links.sparse[PTRN_B]->grid[0].ex +
				domain->links.sparse[PTRN_B]->grid[0].by +
				domain->links.sparse[PTRN_B]->grid[0].ey +
				domain->links.sparse[PTRN_B]->grid[0].bs);

			// Edge (X, Y, 1) subdomain.			
			// Link FRONT edge to the BACK edge
			// of the FRONT neighbour.
			GRID(0, 0, nx, 0, 0, ny, 0, 1, es, 1, 0, 0, ns - 2 * es);
			GRID(1, 0, nx, 0, 0, ny, 0, 2, es, 0, 0, 0, ns - es);
			LINK(PTRN_F, domain->grid[0].es * nes,
				domain->links.sparse[PTRN_F]->grid[0].bx +
				domain->links.sparse[PTRN_F]->grid[0].ex +
				domain->links.sparse[PTRN_F]->grid[0].by +
				domain->links.sparse[PTRN_F]->grid[0].ey);
			
			assert(idomain <= nsubdomains);
		}
	}
}

