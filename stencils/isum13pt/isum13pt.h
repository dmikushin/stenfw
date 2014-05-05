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

#ifndef isum13pt_H
#define isum13pt_H

#include "config.h"

#ifndef __cplusplus
int isum13pt_cpu(int nx, int ny, int ns,
	integer w0[][ny][nx], integer w1[][ny][nx], integer w2[][ny][nx]);
#endif

#ifdef __cplusplus
extern "C"
#endif
int isum13pt_gpu(int nx, int ny, int ns,
	integer* w0, integer* w1, integer* w2);

#endif // isum13pt_H

