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

void grid_gather(
	struct grid_domain_t* dst, struct grid_domain_t* src,
	int igrid, int layout)
{
	struct grid_domain_t* parent = src->parent;
	
	int szelem = dst->szelem;
	
	parent->narrays = dst->narrays;
	parent->szelem = szelem;

	// Set arrays number and size of arrays elements
	// equal for all source domains.
	for (int i = 0; i < parent->nsubdomains; i++)
	{
		src[i].narrays = dst->narrays;
		src[i].szelem = szelem;
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
		// Set data for inner (non-boundary) subdomains.
		int sv = sx * sy * ss, iv = 0;
		size_t offset = 0;
		for (int is = 0, offmuls = 0; is < ss; is++)
		{
			for (int iy = 0, offmuly = 0; iy < sy; iy++)
			{
				for (int ix = 0; ix < sx; ix++, iv++)
				{
					struct grid_domain_t* domain = src + iv;
				
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
					
					size_t loffset = szelem *
						(sbx + sby * (sbx + snx + sex) +
						       sbs * (sbx + snx + sex) *
							     (sby + sny + sey));

					struct grid_domain_t dcpy = *dst;
					dcpy.arrays = dst->arrays;
					dcpy.narrays = dst->narrays;
					dcpy.offset = offset;
					dcpy.grid[0].nx = bbx + bnx + bex;
					dcpy.grid[0].ny = bby + bny + bey;
					dcpy.grid[0].ns = bbs + bns + bes;
			
					struct grid_domain_t scpy = *domain;
					scpy.arrays = domain->arrays;
					scpy.narrays = dst->narrays;
					scpy.offset = szelem *
						(sbx + sby * (sbx + snx + sex) +
						       sbs * (sbx + snx + sex) *
							     (sby + sny + sey));
					scpy.grid[0].nx = bsnx;
					scpy.grid[0].ny = bsny;
					scpy.grid[0].ns = bsns;

					domain->gather_memcpy(
						szelem * snx, sny, sns, &dcpy, &scpy);
				
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
		int nsubdomains = parent->nsubdomains;

		// Set data for inner (non-boundary) subdomains.
		for (int iv = 0; iv < nsubdomains; iv++)
		{
			struct grid_domain_t* domain = src + iv;

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

			size_t offset = szelem * (ix +
				iy * (bx + nx + ex) +
				is * (bx + nx + ex) * (by + ny + ey));
			
			size_t loffset = szelem *
				(sbx + sby * (sbx + snx + sex) +
				       sbs * (sbx + snx + sex) *
					     (sby + sny + sey));

			struct grid_domain_t dcpy = *dst;
			dcpy.arrays = dst->arrays;
			dcpy.narrays = dst->narrays;
			dcpy.offset = offset;
			dcpy.grid[0].nx = bbx + bnx + bex;
			dcpy.grid[0].ny = bby + bny + bey;
			dcpy.grid[0].ns = bbs + bns + bes;
	
			struct grid_domain_t scpy = *domain;
			scpy.arrays = domain->arrays;
			scpy.narrays = dst->narrays;
			scpy.offset = szelem *
				(sbx + sby * (sbx + snx + sex) +
				       sbs * (sbx + snx + sex) *
					     (sby + sny + sey));
			scpy.grid[0].nx = bsnx;
			scpy.grid[0].ny = bsny;
			scpy.grid[0].ns = bsns;

			domain->gather_memcpy(
				szelem * snx, sny, sns, &dcpy, &scpy);
		}
	}
	
	return;
}

