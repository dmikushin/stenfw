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

#define SET_OVERLAPS(bxv, exv, byv, eyv, bsv, esv) \
{ \
	struct grid_domain_t* domain = domains + iv; \
	domain->grid[igrid].bx = bxv; domain->grid[igrid].ex = exv; \
	domain->grid[igrid].by = byv; domain->grid[igrid].ey = eyv; \
	domain->grid[igrid].bs = bsv; domain->grid[igrid].es = esv; \
}

// Set normalized domains overlaps (0 or 1).
void grid_set_overlaps(
	struct grid_domain_t* domains,
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

	// For each domain set the local data overhead.
	{
		int iv = 0;
		
		SET_OVERLAPS(0, ex, 0, ey, 0, es);
		iv += incx;

		for (int ix = 1; ix < sx - 1; ix++, iv++)
		{
			SET_OVERLAPS(bx, ex, 0, ey, 0, es);
		}

		SET_OVERLAPS(bx, 0, 0, ey, 0, es);
		iv += incy;

		for (int iy = 1; iy < sy - 1; iy++)
		{
			SET_OVERLAPS(0, ex, by, ey, 0, es);
			iv += incx;

			for (int ix = 1; ix < sx - 1; ix++, iv++)
			{
				SET_OVERLAPS(bx, ex, by, ey, 0, es);
			}

			SET_OVERLAPS(bx, 0, by, ey, 0, es);
			iv++;
		}

		SET_OVERLAPS(0, ex, by, 0, 0, es);
		iv += incx;

		for (int ix = 1; ix < sx - 1; ix++, iv++)
		{
			SET_OVERLAPS(bx, ex, by, 0, 0, es);
		}

		SET_OVERLAPS(bx, 0, by, 0, 0, es);
		iv += incs;

		for (int is = 1; is < ss - 1; is++)
		{
			SET_OVERLAPS(0, ex, 0, ey, bs, es);
			iv += incx;

			for (int ix = 1; ix < sx - 1; ix++, iv++)
			{
				SET_OVERLAPS(bx, ex, 0, ey, bs, es);
			}

			SET_OVERLAPS(bx, 0, 0, ey, bs, es);
			iv += incy;

			for (int iy = 1; iy < sy - 1; iy++)
			{
				SET_OVERLAPS(0, ex, by, ey, bs, es);
				iv += incx;

				for (int ix = 1; ix < sx - 1; ix++, iv++)
				{
					SET_OVERLAPS(bx, ex, by, ey, bs, es);
				}
				
				SET_OVERLAPS(bx, 0, by, ey, bs, es);
				iv++;
			}

			SET_OVERLAPS(0, ex, by, 0, bs, es);
			iv += incx;

			for (int ix = 1; ix < sx - 1; ix++, iv++)
			{
				SET_OVERLAPS(bx, ex, by, 0, bs, es);
			}

			SET_OVERLAPS(bx, 0, by, 0, bs, es);
			iv++;
		}

		SET_OVERLAPS(0, ex, 0, ey, bs, 0);
		iv += incx;

		for (int ix = 1; ix < sx - 1; ix++, iv++)
		{
			SET_OVERLAPS(bx, ex, 0, ey, bs, 0);
		}

		SET_OVERLAPS(bx, 0, 0, ey, bs, 0);
		iv += incy;

		for (int iy = 1; iy < sy - 1; iy++)
		{
			SET_OVERLAPS(0, ex, by, ey, bs, 0);
			iv += incx;

			for (int ix = 1; ix < sx - 1; ix++, iv++)
			{
				SET_OVERLAPS(bx, ex, by, ey, bs, 0);
			}

			SET_OVERLAPS(bx, 0, by, ey, bs, 0);
			iv++;
		}

		SET_OVERLAPS(0, ex, by, 0, bs, 0);
		iv += incx;

		for (int ix = 1; ix < sx - 1; ix++, iv++)
		{
			SET_OVERLAPS(bx, ex, by, 0, bs, 0);
		}

		SET_OVERLAPS(bx, 0, by, 0, bs, 0);
		iv++;

		assert(iv == sx * sy * ss);
	}
}

