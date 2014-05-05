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

// Perform consistency checks.
void grid_check_valid(
	struct grid_domain_t* domains, int ndomains)
{
	int all_null = 1;
	for (int iv = 0; iv < ndomains; iv++)
	{
		struct grid_domain_t* domain = domains + iv;

		int nsubdomains = domain->nsubdomains;
		for (int i = 0; i < nsubdomains; i++)
		{
			struct grid_domain_t* subdomain1 = domain->subdomains + i;

			// Positive relative poistion indexes.
			assert(subdomain1->grid[0].ix >= 0);
			assert(subdomain1->grid[0].iy >= 0);
			assert(subdomain1->grid[0].is >= 0);

			assert(subdomain1->grid[1].ix >= 0);
			assert(subdomain1->grid[1].iy >= 0);
			assert(subdomain1->grid[1].is >= 0);
		
			// For each domain every subdomain must be linked
			// with domain's neighbour and vise versa.
			struct grid_domain_t* subdomain2 =
				*(subdomain1->links.dense[0]);
			if (subdomain2)
			{
				assert(subdomain1->grid[0].extsize ==
					subdomain2->grid[0].extsize);
				assert(subdomain1 == *(subdomain2->links.dense[0]));
				all_null = 0;
			}
		}
	}
	
	if (ndomains > 1) assert(!all_null);
}

