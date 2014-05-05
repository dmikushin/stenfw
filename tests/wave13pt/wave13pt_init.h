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

#ifndef WAVE7PT_INIT_H
#define WAVE7PT_INIT_H

#include "test_init.h"
#include "wave13pt.h"

// Create a data file in UCAR Vapor format.
void test_create_vapor_vdf(struct test_config_t* t);

// Write step variables into the previously created
// UCAR Vapor format data file.
void test_write_vapor_vdf(struct test_config_t* t, int iarray, int itime);

#endif // WAVE7PT_INIT_H

