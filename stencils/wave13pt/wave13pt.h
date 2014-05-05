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

#ifndef WAVE7PT_H
#define WAVE7PT_H

#include "config.h"

#ifndef __cplusplus
int wave13pt_cpu(int nx, int ny, int ns,
	const real c0, const real c1, const real c2,
	real w0[][ny][nx], real w1[][ny][nx], real w2[][ny][nx]);
#endif

#ifdef __cplusplus
extern "C"
#endif
int wave13pt_gpu(int nx, int ny, int ns,
	const real c0, const real c1, const real c2,
	real* w0, real* w1, real* w2);

#endif // WAVE7PT_H

