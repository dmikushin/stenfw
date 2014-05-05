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
#include <stdio.h>
#include <stdlib.h>

// Build a random integer array.
int genirand(int n, integer* irand /** OUT **/)
{
	double drandmax = (double)RAND_MAX;
	for (int i = 0; i < n; i++)
		irand[i] = (integer)round(rand() / drandmax);
	
	return 0;
}

