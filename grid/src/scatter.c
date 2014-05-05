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

void grid_scatter(
	struct grid_domain_t* dst, struct grid_domain_t* src,
	int igrid, int layout)
{
	struct grid_domain_t* parent = dst->parent;
	
	int szelem = src->szelem;
	
	parent->narrays = src->narrays;
	parent->szelem = szelem;

	// Set arrays number and size of arrays elements
	// equal for all destination domains.
	for (int i = 0; i < parent->nsubdomains; i++)
	{
		dst[i].narrays = src->narrays;
		dst[i].szelem = szelem;
	}

	int nx = parent->grid[igrid].nx;
	int ny = parent->grid[igrid].ny;
	int ns = parent->grid[igrid].ns;
	
	int bx = parent->grid[igrid].bx;
	int by = parent->grid[igrid].by;
	int bs = parent->grid[igrid].bs;

	int ex = parent->grid[igrid].ex;
	int ey = parent->grid[igrid].ey;
	int es = parent->grid[igrid].es;
	
	int sx = parent->sx, sy = parent->sy, ss = parent->ss;
	
	int nsubdomains = parent->nsubdomains;
	
	// Region dimensions in bytes.
	size_t bnx = nx * szelem, bny = ny, bns = ns;
	size_t bbx = bx * szelem, bby = by, bbs = bs;
	size_t bex = ex * szelem, bey = ey, bes = es;

	// With automatic layout data offsets are generated,
	// presuming subdomains should go one after another,
	// without spacings.
	if (layout == LAYOUT_MODE_AUTO)
	{
		int sv = sx * sy * ss;

		// Set data for inner (non-boundary) subdomains.
		int iv = 0;
		for (int is = 0, offset = 0, offmuls = 0; is < ss; is++)
		{
			for (int iy = 0, offmuly = 0; iy < sy; iy++)
			{
				for (int ix = 0; ix < sx; ix++, iv++)
				{
					struct grid_domain_t* domain = dst + iv;

					offmuly = domain->grid[igrid].ny - 1;
					offmuls = domain->grid[igrid].ns - 1;

					int snx = domain->grid[igrid].nx;
					int sny = domain->grid[igrid].ny;
					int sns = domain->grid[igrid].ns;
				
					int sbx = domain->grid[igrid].bx;
					int sex = domain->grid[igrid].ex;
					
					int sby = domain->grid[igrid].by;
					int sey = domain->grid[igrid].ey;
					
					int sbs = domain->grid[igrid].bs;
					int ses = domain->grid[igrid].es;
				
					size_t bsnx = szelem * (sbx + snx + sex);
					size_t bsny = (sby + sny + sey);
					size_t bsns = (sbs + sns + ses);

					size_t loffset = offset - szelem *
						(sbx + sby * nx + sbs * nx * ny);
					
					assert(loffset >= 0);

					struct grid_domain_t dcpy = *domain;
					dcpy.arrays = domain->arrays;
					dcpy.narrays = src->narrays;
					dcpy.offset = 0;
					dcpy.grid[0].nx = bsnx;
					dcpy.grid[0].ny = bsny;
					dcpy.grid[0].ns = bsns;
					
					struct grid_domain_t scpy = *src;
					scpy.arrays = src->arrays;
					scpy.narrays = src->narrays;
					scpy.offset = loffset;
					scpy.grid[0].nx = bbx + bnx + bex;
					scpy.grid[0].ny = bby + bny + bey;
					scpy.grid[0].ns = bbs + bns + bes;
						
					domain->scatter_memcpy(
						bsnx, bsny, bsns, &dcpy, &scpy);
				
					offset += domain->grid[igrid].nx * szelem;
				}
			
				offset += nx * offmuly * szelem;
			}

			offset += nx * ny * offmuls * szelem;
		}
	
		assert(iv == sv);

		return;
	}
	
	// With custom layout offsets are generated
	// from the predefined (ix, iy, is) positions
	// given for each subdomain.
	if (layout == LAYOUT_MODE_CUSTOM)
	{
		// Set data for inner (non-boundary) subdomains.
		for (int iv = 0; iv < nsubdomains; iv++)
		{
			struct grid_domain_t* domain = dst + iv;

			int snx = domain->grid[igrid].nx;
			int sny = domain->grid[igrid].ny;
			int sns = domain->grid[igrid].ns;
		
			int sbx = domain->grid[igrid].bx;
			int sex = domain->grid[igrid].ex;

			int sby = domain->grid[igrid].by;
			int sey = domain->grid[igrid].ey;
			
			int sbs = domain->grid[igrid].bs;
			int ses = domain->grid[igrid].es;
		
			size_t bsnx = szelem * (sbx + snx + sex);
			size_t bsny = (sby + sny + sey);
			size_t bsns = (sbs + sns + ses);

			int ix = domain->grid[igrid].ix;
			int iy = domain->grid[igrid].iy;
			int is = domain->grid[igrid].is;

			size_t loffset = szelem * (ix +
				iy * (bx + nx + ex) +
				is * (bx + nx + ex) * (by + ny + ey));
			
			loffset -= szelem * (sbx + (bx + nx + ex) *
				(sby + sbs * (by + ny + ey)));
			
			assert(loffset >= 0);

			struct grid_domain_t dcpy = *domain;
			dcpy.arrays = domain->arrays;
			dcpy.narrays = src->narrays;
			dcpy.offset = 0;
			dcpy.grid[0].nx = bsnx;
			dcpy.grid[0].ny = bsny;
			dcpy.grid[0].ns = bsns;
			
			struct grid_domain_t scpy = *src;
			scpy.arrays = src->arrays;
			scpy.narrays = src->narrays;
			scpy.offset = loffset;
			scpy.grid[0].nx = bbx + bnx + bex;
			scpy.grid[0].ny = bby + bny + bey;
			scpy.grid[0].ns = bbs + bns + bes;
				
			domain->scatter_memcpy(
				bsnx, bsny, bsns, &dcpy, &scpy);
		}
	}

	return;
}

