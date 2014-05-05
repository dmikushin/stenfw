/*
 * STENFW – a stencil framework for compilers benchmarking.
 *
 * Copyright (C) 2012 Dmitry Mikushin, University of Lugano
 *
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "gensine3d.h"

#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327
#endif

// Build 3d Sine function.
int gensine3d(int nx, int ny, int ns, int ox, int oy, int os,
	real dx, real dy, real ds, real sine[][ny][nx] /** OUT **/)
{
	for (int k = 0; k < ns; k++)
		for (int j = 0; j < ny; j++)
			for (int i = 0; i < nx; i++)
				sine[k][j][i] = 
					sin(2 * M_PI * ((i + ox) * dx)) *
					sin(2 * M_PI * ((j + oy) * dy)) *
					sin(2 * M_PI * ((k + os) * ds));
	
	return 0;
}

