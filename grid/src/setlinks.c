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

#define SET_LINKS(bx, ex, by, ey, bs, es) \
{ \
	struct grid_domain_t* domain = domains + iv; \
	struct grid_domain_t** dlinks = (domain->links.sparse); \
	dlinks[PTRN_L] = (struct grid_domain_t*)((1 - bx) * (intptr_t)empty); \
	dlinks[PTRN_R] = (struct grid_domain_t*)((1 - ex) * (intptr_t)empty); \
	dlinks[PTRN_D] = (struct grid_domain_t*)((1 - by) * (intptr_t)empty); \
	dlinks[PTRN_U] = (struct grid_domain_t*)((1 - ey) * (intptr_t)empty); \
	dlinks[PTRN_B] = (struct grid_domain_t*)((1 - bs) * (intptr_t)empty); \
	dlinks[PTRN_F] = (struct grid_domain_t*)((1 - es) * (intptr_t)empty); \
\
	char** clinks = (char**)(domain->links.sparse); \
	clinks[PTRN_L] += (bx * (intptr_t)(domains + iv - 1)); \
	clinks[PTRN_R] += (ex * (intptr_t)(domains + iv + 1)); \
	clinks[PTRN_D] += (by * (intptr_t)(domains + iv - sx)); \
	clinks[PTRN_U] += (ey * (intptr_t)(domains + iv + sx)); \
	clinks[PTRN_B] += (bs * (intptr_t)(domains + iv - sx * sy)); \
	clinks[PTRN_F] += (es * (intptr_t)(domains + iv + sx * sy)); \
\
	int iptrn = 0; \
	domain->links.dense[iptrn] = &(dlinks[PTRN_L]); iptrn += bx; \
	domain->links.dense[iptrn] = &(dlinks[PTRN_R]); iptrn += ex; \
	domain->links.dense[iptrn] = &(dlinks[PTRN_D]); iptrn += by; \
	domain->links.dense[iptrn] = &(dlinks[PTRN_U]); iptrn += by; \
	domain->links.dense[iptrn] = &(dlinks[PTRN_B]); iptrn += bs; \
	domain->links.dense[iptrn] = &(dlinks[PTRN_F]); \
	domain->links.ndense = iptrn; \
}

// Set links between neighboring grid domains.
void grid_set_links(
	struct grid_domain_t* domains,
	struct grid_domain_t* empty,
	int igrid, int sx, int sy, int ss,
	int bx, int ex, int by, int ey, int bs, int es)
{
	// XXX: Handle a special case:
	// when domains grid dimension = 1
	// we cannot put any overhead, other than zero.
	int incx = 1, incy = 1, incs = 1;
	if (sx == 1) { bx = 0; ex = 0; incx = 0; }
	if (sy == 1) { by = 0; ey = 0; incy = 1 - sx; }
	if (ss == 1) { bs = 0; es = 0; incs = 1 - sx * sy; }

	// Normalized versions of fictive boundaries
	// thincknesses.
	bx = bx ? 1 : 0; by = by ? 1 : 0; bs = bs ? 1 : 0;
	ex = ex ? 1 : 0; ey = ey ? 1 : 0; es = es ? 1 : 0;

	// For each domain set links.
	{
		int iv = 0;
		
		SET_LINKS(0, ex, 0, ey, 0, es);
		iv += incx;

		for (int ix = 1; ix < sx - 1; ix++, iv++)
		{
			SET_LINKS(bx, ex, 0, ey, 0, es);
		}

		SET_LINKS(bx, 0, 0, ey, 0, es);
		iv += incy;

		for (int iy = 1; iy < sy - 1; iy++)
		{
			SET_LINKS(0, ex, by, ey, 0, es);
			iv += incx;

			for (int ix = 1; ix < sx - 1; ix++, iv++)
			{
				SET_LINKS(bx, ex, by, ey, 0, es);
			}

			SET_LINKS(bx, 0, by, ey, 0, es);
			iv++;
		}

		SET_LINKS(0, ex, by, 0, 0, es);
		iv += incx;

		for (int ix = 1; ix < sx - 1; ix++, iv++)
		{
			SET_LINKS(bx, ex, by, 0, 0, es);
		}

		SET_LINKS(bx, 0, by, 0, 0, es);
		iv += incs;

		for (int is = 1; is < ss - 1; is++)
		{
			SET_LINKS(0, ex, 0, ey, bs, es);
			iv += incx;

			for (int ix = 1; ix < sx - 1; ix++, iv++)
			{
				SET_LINKS(bx, ex, 0, ey, bs, es);
			}

			SET_LINKS(bx, 0, 0, ey, bs, es);
			iv += incy;

			for (int iy = 1; iy < sy - 1; iy++)
			{
				SET_LINKS(0, ex, by, ey, bs, es);
				iv += incx;

				for (int ix = 1; ix < sx - 1; ix++, iv++)
				{
					SET_LINKS(bx, ex, by, ey, bs, es);
				}
				
				SET_LINKS(bx, 0, by, ey, bs, es);
				iv++;
			}

			SET_LINKS(0, ex, by, 0, bs, es);
			iv += incx;

			for (int ix = 1; ix < sx - 1; ix++, iv++)
			{
				SET_LINKS(bx, ex, by, 0, bs, es);
			}

			SET_LINKS(bx, 0, by, 0, bs, es);
			iv++;
		}

		SET_LINKS(0, ex, 0, ey, bs, 0);
		iv += incx;

		for (int ix = 1; ix < sx - 1; ix++, iv++)
		{
			SET_LINKS(bx, ex, 0, ey, bs, 0);
		}

		SET_LINKS(bx, 0, 0, ey, bs, 0);
		iv += incy;

		for (int iy = 1; iy < sy - 1; iy++)
		{
			SET_LINKS(0, ex, by, ey, bs, 0);
			iv += incx;

			for (int ix = 1; ix < sx - 1; ix++, iv++)
			{
				SET_LINKS(bx, ex, by, ey, bs, 0);
			}

			SET_LINKS(bx, 0, by, ey, bs, 0);
			iv++;
		}

		SET_LINKS(0, ex, by, 0, bs, 0);
		iv += incx;

		for (int ix = 1; ix < sx - 1; ix++, iv++)
		{
			SET_LINKS(bx, ex, by, 0, bs, 0);
		}

		SET_LINKS(bx, 0, by, 0, bs, 0);
		iv++;

		assert(iv == sx * sy * ss);
	}
}

