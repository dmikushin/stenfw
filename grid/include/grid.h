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

#ifndef GRID_H
#define GRID_H

/**
 * The grid domain layout mode.
 * With automatic layout data offsets are generated,
 * presuming subdomains should go one after another,
 * without spacings.
 */
#define LAYOUT_MODE_AUTO   0

/**
 * The grid domain layout mode.
 * With custom layout offsets are generated
 * from the predefined (ix, iy, is) positions
 * given for each subdomain.
 */
#define LAYOUT_MODE_CUSTOM 1

/**
 * The grid engine primary grid index.
 */
#define GRID_PRIMARY	0

/**
 * The grid engine secondary grid index.
 */
#define GRID_SECONDARY	1

#define PTRN_C 0
#define PTRN_L 1
#define PTRN_R 2
#define PTRN_D 3
#define PTRN_U 4
#define PTRN_B 5
#define PTRN_F 6
#define PTRN_N 7

#include <stddef.h> // size_t

struct grid_domain_t;

/**
 * The grid pattern. Defines dense and sparse
 * representation of domain neighborhood.
 */
struct grid_pattern_t
{
	struct grid_domain_t* sparse[PTRN_N];
	struct grid_domain_t** dense[PTRN_N];
	
	// The number of non-null elements in pattern.
	int ndense;
};

/**
 * The grid configuration.
 */
struct grid_t
{
	int nx, ny, ns;
	int bx, by, bs;
	int ex, ey, es;

	int size, extsize;

	// The index of domain relative to its parent.
	int ix, iy, is;
	
	// The offset of domain grid points, relative to the grid
	// of the parent domain.
	int ox, oy, os;
};

typedef void (*memcpy_func_t)(int nx, int ny, int ns,
	struct grid_domain_t* dst, struct grid_domain_t* src);

/**
 * The domain configuration.
 */
struct grid_domain_t
{
	// Primary and secondary domain grids.
	struct grid_t grid[2];

	char** arrays;
	int narrays, szelem, offset;

	// A pointer to access the parent domain
	// configuration. Top level domains should have
	// NULL parent.
	struct grid_domain_t* parent;

	// A pointer to access the nested domains -
	// the domains of inner grid decomposition. Domains
	// without nested subdomains should have NULL subdomains.
	struct grid_domain_t* subdomains;
	int nsubdomains;

	// Nested domains subgrid.
	int sx, sy, ss;

	// Indirectly linked domains array.
	struct grid_pattern_t links;
	
	// External functions to copy domain data
	// in scatter and gather operations.
	memcpy_func_t scatter_memcpy, gather_memcpy;
	
	// The user-defined data.
	void* data;
};

/**
 * Create the simple decomposition grid.
 * @param nx - The global grid X dimension
 * @param ny - The global grid Y dimension
 * @param ns - The global grid Z dimension
 * @param sx - The number of subdomains by X dimension
 * @param sy - The number of subdomains by Y dimension
 * @param ss - The number of subdomains by Z dimension
 * @param bx - The subdomain left overlap thickness by X dimension
 * @param ex - The subdomain right overlap thickness by X dimension
 * @param by - The subdomain upper overlap thickness by Y dimension
 * @param ey - The subdomain lower overlap thickness by Y dimension
 * @param bs - The subdomain top overlap thickness by Z dimension
 * @param es - The subdomain bottom overlap thickness by Z dimension
 * @return The configured domains array.
 */
struct grid_domain_t* grid_init_simple(
	int nx, int ny, int ns,	int sx, int sy, int ss,
	int bx, int ex, int by, int ey, int bs, int es);

/**
 * Allocate space and perform initial setup of domains
 * configuration. Memory allocation includes space for
 * nested subdomains and links: parent domain + 
 * nsubdomains + PTRN_N nested domains in each subdomain.
 * @param nx - The global grid X dimension
 * @param ny - The global grid Y dimension
 * @param ns - The global grid Z dimension
 * @param sx - The number of subdomains by X dimension
 * @param sy - The number of subdomains by Y dimension
 * @param ss - The number of subdomains by Z dimension
 * @param bx - The subdomain left overlap thickness by X dimension
 * @param ex - The subdomain right overlap thickness by X dimension
 * @param by - The subdomain upper overlap thickness by Y dimension
 * @param ey - The subdomain lower overlap thickness by Y dimension
 * @param bs - The subdomain top overlap thickness by Z dimension
 * @param es - The subdomain bottom overlap thickness by Z dimension
 * @return The configured domains array.
 */
struct grid_domain_t* grid_allocate(
	int nx, int ny, int ns, int sx, int sy, int ss,
	int bx, int ex, int by, int ey, int bs, int es);

/**
 * Decompose grid into domains, trying to distribute
 * the remainder across domains, i.e. get balanced distribution
 * @param nx - The global grid X dimension
 * @param ny - The global grid Y dimension
 * @param ns - The global grid Z dimension
 * @param sx - The number of subdomains by X dimension
 * @param sy - The number of subdomains by Y dimension
 * @param ss - The number of subdomains by Z dimension
 * @param bx - The subdomain left overlap thickness by X dimension
 * @param ex - The subdomain right overlap thickness by X dimension
 * @param by - The subdomain upper overlap thickness by Y dimension
 * @param ey - The subdomain lower overlap thickness by Y dimension
 * @param bs - The subdomain top overlap thickness by Z dimension
 * @param es - The subdomain bottom overlap thickness by Z dimension
 * @param domains - The domains array to setup
 */
void grid_decompose_balanced(
	struct grid_domain_t* domains,
	int nx, int ny, int ns, int sx, int sy, int ss,
	int bx, int ex, int by, int ey, int bs, int es);

/**
 * Set normalized domains overlaps (0 or 1).
 * @param igrid - The index of grid to set (primary/secondary).
 * @param sx - The number of subdomains by X dimension
 * @param sy - The number of subdomains by Y dimension
 * @param ss - The number of subdomains by Z dimension
 * @param bx - The subdomain left overlap thickness by X dimension
 * @param ex - The subdomain right overlap thickness by X dimension
 * @param by - The subdomain upper overlap thickness by Y dimension
 * @param ey - The subdomain lower overlap thickness by Y dimension
 * @param bs - The subdomain top overlap thickness by Z dimension
 * @param es - The subdomain bottom overlap thickness by Z dimension
 * @param domains - The domains array to setup
 */
void grid_set_overlaps(
	struct grid_domain_t* domains,
	int igrid, int sx, int sy, int ss,
	int bx, int ex, int by, int ey, int bs, int es);

/**
 * Set links between neighboring grid domains.
 * @param iempty - Use NULL or fictive empty domain
 * in place of absent links (0 or 1)
 * @param igrid - The index of grid to set (primary/secondary).
 * @param sx - The number of subdomains by X dimension
 * @param sy - The number of subdomains by Y dimension
 * @param ss - The number of subdomains by Z dimension
 * @param bx - The subdomain left overlap thickness by X dimension
 * @param ex - The subdomain right overlap thickness by X dimension
 * @param by - The subdomain upper overlap thickness by Y dimension
 * @param ey - The subdomain lower overlap thickness by Y dimension
 * @param bs - The subdomain top overlap thickness by Z dimension
 * @param es - The subdomain bottom overlap thickness by Z dimension
 * @param domains - The domains array to setup
 */
void grid_set_links(
	struct grid_domain_t* domains,
	struct grid_domain_t* empty, int igrid, int sx, int sy, int ss,
	int bx, int ex, int by, int ey, int bs, int es);

/**
 * Calculate grid domains sizes.
 * @param domains - The array of domains to calculate sizes for
 * @param ndomains - The domains array length
 */
void grid_set_sizes(
	struct grid_domain_t* domains, int ndomains);

/**
 * Perform consistency checks.
 * @param domains - The array of domains
 * @param ndomains - The domains array length
 */
void grid_check_valid(
	struct grid_domain_t* domains, int ndomains);

/**
 * Set domains edges subgrid.
 * @param sx - The number of subdomains by X dimension
 * @param sy - The number of subdomains by Y dimension
 * @param ss - The number of subdomains by Z dimension
 * @param bx - The subdomain left overlap thickness by X dimension
 * @param ex - The subdomain right overlap thickness by X dimension
 * @param by - The subdomain upper overlap thickness by Y dimension
 * @param ey - The subdomain lower overlap thickness by Y dimension
 * @param bs - The subdomain top overlap thickness by Z dimension
 * @param es - The subdomain bottom overlap thickness by Z dimension
 * @param domains - The array of domains
 * @param ndomains - The domains array length
 */
void grid_set_edges(
	struct grid_domain_t* domains,
	int sx, int sy, int ss,
	int bx, int ex, int by, int ey, int bs, int es);

/**
 * Multiply the grid domains overlaps by the specified values.
 * @param sx - The number of subdomains by X dimension
 * @param sy - The number of subdomains by Y dimension
 * @param ss - The number of subdomains by Z dimension
 * @param bx - The subdomain left overlap thickness by X dimension
 * @param ex - The subdomain right overlap thickness by X dimension
 * @param by - The subdomain upper overlap thickness by Y dimension
 * @param ey - The subdomain lower overlap thickness by Y dimension
 * @param bs - The subdomain top overlap thickness by Z dimension
 * @param es - The subdomain bottom overlap thickness by Z dimension
 * @param domains - The array of domains to calculate sizes for
 * @param ndomains - The domains array length
 */
void grid_overlaps_multiply(
	struct grid_domain_t* domains,
	int nx, int ny, int ns, int sx, int sy, int ss,
	int bx, int ex, int by, int ey, int bs, int es);

/**
 * Decompose arrays given for the specified global grids
 * into continuous subdomains on subgrids selected according
 * to the grid mapping parameters.
 * @param domains - The configured grid domains
 * @param igrid - The index of grid to use (each domain has two grids -
 * primary and secondary)
 * @param arrays - The data arrays to be decomposed into subdomains
 * @param narrays - The number of data arrays
 * @param szelem - The size of data element
 * @param layout - The grid domain layout mode
 * @param scratch - The scratch space buffer
 */
void grid_scatter(
	struct grid_domain_t* dst, struct grid_domain_t* src,
	int igrid, int layout);

/**
 * Compose continuous arrays given on distributed subdomains
 * into global grid space using mapping parameters specified
 * by decomposition descriptor.
 * @param domains - The configured grid domains
 * @param igrid - The index of grid to use (each domain has two grids -
 * primary and secondary)
 * @param arrays - The data arrays to be decomposed into subdomains
 * @param narrays - The number of data arrays
 * @param szelem - The size of data element
 * @param layout - The grid domain layout mode
 * @param scratch - The scratch space buffer
 */
void grid_gather(
	struct grid_domain_t* dst, struct grid_domain_t* src,
	int igrid, int layout);

/**
 * Copy 3d source subdomain to destanation domain without buffering,
 * using memcpy function.
 * @param nx - The subdomain X dimension
 * @param ny - The subdomain Y dimension
 * @param ns - The subdomain Z dimension
 * @param dst - The destination subdomain data pointer
 */
void grid_subcpy(int nx, int ny, int ns,
	struct grid_domain_t* dst, struct grid_domain_t* src);

#endif // GRID_H

