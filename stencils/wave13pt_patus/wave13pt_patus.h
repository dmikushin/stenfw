#ifndef WAVE7PT_PATUS_H
#define WAVE7PT_PATUS_H

#include "config.h"

int wave13pt_patus_avx(int nx, int ny, int ns,
	const real c0, const real c1, const real c2,
	real w0[][ny][nx], real w1[][ny][nx], real w2[][ny][nx]);

int wave13pt_patus_avxfma4(int nx, int ny, int ns,
	const real c0, const real c1, const real c2,
	real w0[][ny][nx], real w1[][ny][nx], real w2[][ny][nx]);

#endif // WAVE7PT_PATUS_H

