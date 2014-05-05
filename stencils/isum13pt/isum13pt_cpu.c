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

#include "isum13pt.h"

int isum13pt_cpu(int nx, int ny, int ns,
	integer w0[][ny][nx], integer w1[][ny][nx], integer w2[][ny][nx])
{
	#pragma omp parallel for
	for (int k = 2; k < ns - 2; k++)
		for (int j = 2; j < ny - 2; j++)
			for (int i = 2; i < nx - 2; i++)
				w2[k][j][i] =  w1[k][j][i] + w0[k][j][i] +

						w0[k][j][i+1] + w0[k][j][i-1]  +
						w0[k][j+1][i] + w0[k][j-1][i]  +
						w0[k+1][j][i] + w0[k-1][j][i]  +

						w0[k][j][i+2] + w0[k][j][i-2]  +
						w0[k][j+2][i] + w0[k][j-2][i]  +
						w0[k+2][j][i] + w0[k-2][j][i]  +

						w1[k][j][i+1] + w1[k][j][i-1]  +
						w1[k][j+1][i] + w1[k][j-1][i]  +
						w1[k+1][j][i] + w1[k-1][j][i]  +

						w1[k][j][i+2] + w1[k][j][i-2]  +
						w1[k][j+2][i] + w1[k][j-2][i]  +
						w1[k+2][j][i] + w1[k-2][j][i];
	
	return 0;
}

