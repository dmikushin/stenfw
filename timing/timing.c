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

#include "timing.h"

#include <stdio.h>
#include <sys/time.h>

#ifdef CLOCK_GETTIME_NOT_IMPLEMENTED
int clock_gettime(int id, struct timespec* t)
{
	struct timeval now;
	int rv = gettimeofday(&now, NULL);
	if (rv) return rv;
	t->tv_sec  = now.tv_sec;
	t->tv_nsec = now.tv_usec * 1000;
	return 0;
}
#endif

// Get the built-in timer value.
void stenfw_get_time(struct timespec* t)
{
	clock_gettime(CLOCKID, t);
}

// Get the built-in timer measured values difference.
double stenfw_get_time_diff(
	struct timespec t1, struct timespec t2)
{
	long tv_sec = t2.tv_sec - t1.tv_sec;
	long tv_nsec = t2.tv_nsec - t1.tv_nsec;
	
	if (t2.tv_nsec < t1.tv_nsec)
	{
		tv_sec--;
		tv_nsec = (1000000000 - t1.tv_nsec) + t2.tv_nsec;
	}
	
	return (double)0.000000001 * tv_nsec + tv_sec;
}

// Print the built-in timer measured values difference.
void stenfw_print_time_diff(
	struct timespec t1, struct timespec t2)
{
	long tv_sec = t2.tv_sec - t1.tv_sec;
	long tv_nsec = t2.tv_nsec - t1.tv_nsec;
	
	if (t2.tv_nsec < t1.tv_nsec)
	{
		tv_sec--;
		tv_nsec = (1000000000 - t1.tv_nsec) + t2.tv_nsec;
	}
	printf("%ld.%09ld", tv_sec, tv_nsec);
}

