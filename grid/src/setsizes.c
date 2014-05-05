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

// Calculate domains sizes.
void grid_set_sizes(
	struct grid_domain_t* domains, int ndomains)
{
	// Calculate domains sizes.
	for (int iv = 0; iv < ndomains; iv++)
	{
		struct grid_domain_t* domain = domains + iv;

		// Caluclate normal and extended domain size
		// for the domain primary grid.
		domain->grid[0].size = 
			domain->grid[0].nx * domain->grid[0].ny * domain->grid[0].ns;
		domain->grid[0].extsize = 
			(domain->grid[0].bx + domain->grid[0].nx + domain->grid[0].ex) *
			(domain->grid[0].by + domain->grid[0].ny + domain->grid[0].ey) *
			(domain->grid[0].bs + domain->grid[0].ns + domain->grid[0].es);

		// Caluclate normal and extended domain size
		// for the domain secondary grid.
		domain->grid[1].size = 
			domain->grid[1].nx * domain->grid[1].ny * domain->grid[1].ns;
		domain->grid[1].extsize = 
			(domain->grid[1].bx + domain->grid[1].nx + domain->grid[1].ex) *
			(domain->grid[1].by + domain->grid[1].ny + domain->grid[1].ey) *
			(domain->grid[1].bs + domain->grid[1].ns + domain->grid[1].es);

		int nsubdomains = domain->nsubdomains;
		for (int i = 0; i < nsubdomains; i++)
		{
			struct grid_domain_t* sub = domain->subdomains + i;

			// Caluclate normal and extended domain size
			// for the subdomain primary grid.		
			sub->grid[0].size =
				sub->grid[0].nx * sub->grid[0].ny * sub->grid[0].ns;
			sub->grid[0].extsize =
				(sub->grid[0].bx + sub->grid[0].nx + sub->grid[0].ex) *
				(sub->grid[0].by + sub->grid[0].ny + sub->grid[0].ey) *
				(sub->grid[0].bs + sub->grid[0].ns + sub->grid[0].es);

			// Caluclate normal and extended domain size
			// for the subdomain secondary grid.		
			sub->grid[1].size =
				sub->grid[1].nx * sub->grid[1].ny * sub->grid[1].ns;
			sub->grid[1].extsize =
				(sub->grid[1].bx + sub->grid[1].nx + sub->grid[1].ex) *
				(sub->grid[1].by + sub->grid[1].ny + sub->grid[1].ey) *
				(sub->grid[1].bs + sub->grid[1].ns + sub->grid[1].es);
		}
	}
}

