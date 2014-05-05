//
//      $Id: Lifting1D.h,v 1.8 2010/02/11 18:10:38 alannorton Exp $
//

#ifndef	_Lifting1D_h_
#define	_Lifting1D_h_

#include <iostream>
#include <vapor/EasyThreads.h>
#include <vapor/MyBase.h>
#ifdef WIN32
#pragma warning(disable : 4996)
#endif
namespace VAPoR {

//
//! \class Lifting1D
//! \brief Wrapper for Wim Swelden's Liftpack wavelet transform interface
//! \author John Clyne
//! \version $Revision: 1.8 $
//! \date    $Date: 2010/02/11 18:10:38 $
//!
//! This class provides an interface to the Liftpack wavelet transformation
//! library.
//
template <class Data_T> class Lifting1D : public VetsUtil::MyBase {

public:

 //! maximum number of filter coefficients
 static const int	MAX_FILTER_COEFF = 32;

 //! \param[in] n Number of wavelet filter coefficients. Valid values are
 //! from 1 to Lifting1D::MAX_FILTER_COEFF
 //! \param[in] ntilde Number of wavelet lifting coefficients. Valid values are
 //! from 1 to Lifting1D::MAX_FILTER_COEFF
 //! \param[in] width Number of samples to be transformed
 //
 Lifting1D(
	unsigned int n,		// # wavelet filter coefficents
	unsigned int ntilde,	// # wavelet lifting coefficients
	unsigned int width,	// # of samples to transform
	int normalize = 0
 );

 ~Lifting1D();

 //! Apply forward lifting transform
 //!
 //! Apply forward lifting transform to \p width samples from \p data. The 
 //! transform is performed in place. 
 //! \param[in,out] data Data to transform
 //! \sa InverseTransform(), Lifting1D()
 //
 void	ForwardTransform(Data_T *data, int stride = 1);

 //! Apply inverse lifting transform
 //!
 //! Apply inverse lifting transform to \p width samples from \p data. The 
 //! transform is performed in place. 
 //! \param[in,out] data Data to transform
 //! \sa ForwardTransform(), Lifting1D()
 //
 void	InverseTransform(Data_T *data, int stride = 1);

private:

 static const double    Flt_Epsilon;
 static const double    TINY;
 int	_objInitialized;	// has the obj successfully been initialized?

 unsigned int	n_c;		// # wavelet filter coefficients
 unsigned int	ntilde_c;	// # wavelet lifting coefficients
 unsigned int	max_width_c;
 unsigned int	width_c;
 double	*fwd_filter_c;		// forward transform filter coefficients
 double	*inv_filter_c;		// inverse transform filter coefficients
 double	*fwd_lifting_c;		// forward transform lifting coefficients
 double	*inv_lifting_c;		// inverse transform lifting coefficients
 double _nfactor;			// normalization factor;

 int zero( const double x );

 double neville ( const double *x, const double *f, const int n, const double xx);

 void update_moment(
    double** moment, const double *filter, const int step2,
    const int noGammas, const int len, const int n, const int ntilde
 );

 double** matrix(long nrl, long nrh, long ncl, long nch);

 void free_matrix(
	double **m, long nrl, long nrh, long ncl, long nch 
 );

 void LUdcmp(double **a, int n, int *indx, double *d);
 void LUbksb (double **a, int n, int *indx, double *b);

 void update_lifting(
    double *lifting, const double** moment, const int len,
    const int step2, const int noGammas, const int ntilde
 );

 double	*create_fwd_filter (unsigned int);
 double	*create_inv_filter (unsigned int, const double *);
 double	**create_moment(
    const unsigned int, const unsigned int, const int
 );

 double *create_inv_lifting(double *, unsigned int, unsigned int);

 double *create_fwd_lifting(
    const double *filter, const unsigned int n,
    const unsigned int ntilde, const unsigned int width
 );

 void FLWT1D_Predict(Data_T*, const long, const long, double *, int stride);
 void FLWT1D_Update(Data_T *, const long, const long, const double*, int stride);

 void forward_transform1d_haar(Data_T *data, int width, int stride);

 void inverse_transform1d_haar(Data_T *data, int width, int stride);


};

};



/*
 *  -*- Mode: ANSI C -*-
 *  $Id: Lifting1D.h,v 1.8 2010/02/11 18:10:38 alannorton Exp $
 *  Author: Gabriel Fernandez, Senthil Periaswamy
 *
 *     >>> Fast Lifted Wavelet Transform on the Interval <<<
 *
 *                             .-. 
 *                            / | \
 *                     .-.   / -+- \   .-.
 *                        `-'   |   `-'
 *
 *  Using the "Lifting Scheme" and the "Square 2D" method, the coefficients
 *  of a "Biorthogonal Wavelet" or "Second Generation Wavelet" are found for
 *  a two-dimensional data set. The same scheme is used to apply the inverse
 *  operation.
 */
/* do not edit anything above this line */

/* System header files */
#include <cstdio>
#include <cstdlib>   
#include <cmath>
#include <cstring>   
#include <cerrno>   
#include <cassert>   

using namespace VAPoR;
using namespace VetsUtil;

#ifndef	MAX
#define	MAX(A,B) ((A)>(B)?(A):(B))
#endif


template <class Data_T> const double Lifting1D<Data_T>::TINY = (double) 1.0e-20;

template <class Data_T> Lifting1D<Data_T>::Lifting1D(
	unsigned int	n,
	unsigned int	ntilde,
	unsigned int	width,
	int normalize
) {

	_objInitialized = 0;
	n_c = n;
	ntilde_c = ntilde;
	width_c = width;
	fwd_filter_c = NULL;
	fwd_lifting_c = NULL;
	inv_filter_c = NULL;
	inv_lifting_c = NULL;
	_nfactor = 1.0;

	SetClassName("Lifting1D");


	// compute normalization factor if needed
	if (normalize) {
		int j = 0;

		while (width > 2) {
			width -= (width>>1);
			j++;
		}
		_nfactor = sqrt(pow(2.0, (double) j));
	}

	// Check for haar wavelets which are handled differently
	//
	if (n_c == 1 && ntilde_c == 1) {
		_objInitialized = 1;
		return;
	}


	if (IsOdd(n_c)) {
		SetErrMsg(
			"Invalid # lifting coeffs., n=%d, is odd", 
			n_c
		);
		return;
	}

	if (n_c > MAX_FILTER_COEFF) {
		SetErrMsg(
			"Invalid # of lifting coeffs., n=%d, exceeds max=%d",
			n_c, MAX_FILTER_COEFF
		);
		return;
	}
	if (ntilde_c > MAX_FILTER_COEFF) {
		SetErrMsg(
			"Invalid # of lifting coeffs., ntilde=%d, exceeds max=%d",
			ntilde_c, MAX_FILTER_COEFF
		);
		return;
	}

	int	maxN = MAX(n_c, ntilde_c) - 1;	// max vanishing moments 
	int	maxl = (width_c == 1) ? 0 : (int)LogBaseN ((double)(width_c-1)/maxN, (double) 2.0);

	if (maxl < 1) {
		SetErrMsg(
			"Invalid # of samples, width=%d, less than # moments",
			width
		);
		return;
		
	}
	

	if (! (fwd_filter_c = create_fwd_filter(n_c))) {
		SetErrMsg("malloc() : %s", strerror(errno));
		return;
	}
	if (! (inv_filter_c = create_inv_filter(n_c, fwd_filter_c))) {
		SetErrMsg("malloc() : %s", strerror(errno));
		return;
	}


#ifdef	DEBUGGING
	{
            int Nrows = (n_c>>1) + 1;
            filterPtr = filter_c;
            fprintf (stdout, "\nFilter Coefficients (N=%d)\n", n_c);
            for (row = 0 ; row < Nrows ; row++) {
                fprintf (stdout, "For %d in the left and %d in the right\n[ ",
                         row, N-row);
                for (col = 0 ; col < n_c ; col++)
                    fprintf (stdout, "%.18g ", *(filterPtr++));
                fprintf (stdout, "]\n");
            }
        }
#endif


	fwd_lifting_c = create_fwd_lifting(fwd_filter_c,n_c,ntilde_c,width_c);
	if (! fwd_lifting_c) {
		SetErrMsg("malloc() : %s", strerror(errno));
		return;
	}

	inv_lifting_c = create_inv_lifting(fwd_lifting_c, ntilde_c, width_c);
	if (! inv_lifting_c) {
		SetErrMsg("malloc() : %s", strerror(errno));
		return;
	}


	_objInitialized = 1;
}

template <class Data_T> Lifting1D<Data_T>::~Lifting1D()
{
	if (! _objInitialized) return;

	if (fwd_filter_c) delete [] fwd_filter_c;
	fwd_filter_c = NULL;

	if (inv_filter_c) delete [] inv_filter_c;
	inv_filter_c = NULL;

	if (fwd_lifting_c) delete [] fwd_lifting_c;
	fwd_lifting_c = NULL;

	if (inv_lifting_c) delete [] inv_lifting_c;
	inv_lifting_c = NULL;

	_objInitialized = 0;
}

/* CODE FOR THE IN-PLACE FLWT (INTERPOLATION CASE, N EVEN) */

//
// ForwardTransform function: This is a 1D Forward Fast Lifted Wavelet
//                  Trasform. Since the Lifting Scheme is used, the final
//                  coefficients are organized in-place in the original 
//                  vector Data[]. In file mallat.c there are
//                  functions provided to change this organization to
//                  the Isotropic format originally established by Mallat.
//
template <class Data_T> void   Lifting1D<Data_T>::ForwardTransform(
	Data_T *data, int stride
) {
	if (n_c == 1 && ntilde_c == 1) {
		forward_transform1d_haar(data, width_c, stride);
	}
	else {
		FLWT1D_Predict (data, width_c, n_c, fwd_filter_c, stride);
		FLWT1D_Update (data, width_c, ntilde_c, fwd_lifting_c, stride);
	}

	if (_nfactor != 1.0) {
		// normalize the detail coefficents
		for(unsigned int i=1; i<width_c; i+=2) {
			data[i*stride] /= _nfactor;
		}
	}
}

//
// InverseTransform function: This is a 1D Inverse Fast Lifted Wavelet
//                  Trasform. Since the Lifting Scheme is used, the final
//                  coefficients are organized in-place in the original 
//                  vector Data[]. In file mallat.c there are
//                  functions provided to change this organization to
//                  the Isotropic format originally established by Mallat.
//
template <class Data_T> void Lifting1D<Data_T>::InverseTransform(
	Data_T *data, int stride
) {
	if (_nfactor != 1.0) {
		// un-normalize the detail coefficents
		for(unsigned int i=1; i<width_c; i+=2) {
			data[i*stride] *= _nfactor;
		}
	}

	if (n_c == 1 && ntilde_c == 1) {
		inverse_transform1d_haar(data, width_c, stride);
	}
	else {
		FLWT1D_Update (data, width_c, ntilde_c, inv_lifting_c, stride);
		FLWT1D_Predict (data, width_c, n_c, inv_filter_c, stride);
	}
}

template <class Data_T> const double Lifting1D<Data_T>::Flt_Epsilon =   (double) 6E-8;
template <class Data_T> int Lifting1D<Data_T>::zero( const double x )
{
	const double __negeps_f = -100 * Flt_Epsilon;
	const double __poseps_f = 100 * Flt_Epsilon;

	return ( __negeps_f < x ) && ( x < __poseps_f );
}

/*
 *  -*- Mode: ANSI C -*-
 *  (Polynomial Interpolation)
 *  $Id: Lifting1D.h,v 1.8 2010/02/11 18:10:38 alannorton Exp $
 *  Author: Wim Sweldens, Gabriel Fernandez
 *
 *  Given n points of the form (x,f), this program uses the Neville algorithm
 *  to find the value at xx of the polynomial of degree (n-1) interpolating
 *  the points (x,f).
 *  Ref: Stoer and Bulirsch, Introduction to Numerical Analysis,
 *  Springer-Verlag.
 */

template <class Data_T> double Lifting1D<Data_T>::neville(
	const double *x, const double *f, const int n, const double xx 
) {
	register int i,j;
	double	vy[MAX_FILTER_COEFF];
	double y;

	for ( i=0; i<n; i++ ) {
		vy[i] = f[i];
		for ( j=i-1; j>=0; j--) {
			double den = x[i] - x[j];
			assert(! zero(den));
			vy[j] = vy[j+1] + (vy[j+1] - vy[j]) * (xx - x[i]) / den;
		}
	}
	y = vy[0];

	return y;
}

//
// UpdateMoment function: calculates the integral-moment tuple for the current
//                        level of calculations.
//
template <class Data_T> void Lifting1D<Data_T>::update_moment(
	double** moment,
	const double *filter,
	const int step2,
	const int noGammas, 
	const int len,
	const int n,
	const int ntilde
) {
	int i, j,              /* counters */
	row, col,          /* indices of the matrices */
	idxL, idxG,        /* pointers to Lambda & Gamma coeffs, resp. */
	top1, top2, top3;  /* number of iterations for L<ntilde, L=ntilde, L>ntilde */
	const double	*filterPtr;	// ptr to filter coefficients

	/***************************************/
	/* Update Integral-Moments information */
	/***************************************/

	/* Calculate number of iterations for each case */
	top1 = (n>>1) - 1;                 /* L < ntilde */
	top3 = (n>>1) - IsOdd(len);          /* L > ntilde */
	top2 = noGammas - (top1 + top3);   /* L = ntilde */

	/* Cases where nbr. left Lambdas < nbr. right Lambdas */
	filterPtr = filter;   /* initialize pointer to first row */
	idxG = step2>>1;      /* second coefficient is Gamma */
	for ( row = 1 ; row <= top1 ; row++ ) {
		idxL = 0;   /* first Lambda is always the first coefficient */
		filterPtr += n;   /* go to next filter row */
		for ( col = 0 ; col < n ; col++ ) {
			/* Update (int,mom_1,mom_2,...) */
			for ( j = 0 ; j < ntilde ; j++ ) {
				moment[idxL][j] += filterPtr[col]*moment[idxG][j];
			}
			/* Jump to next Lambda coefficient */
			idxL += step2;
		}
		idxG += step2;   /* go to next Gamma coefficient */
	}

	/* Cases where nbr. left Lambdas = nbr. right Lambdas */
	filterPtr += n;   /* go to last filter row */
	for ( i = 0 ; i < top2 ; i++ ) {
		idxL = i*step2;
		for ( col = 0 ; col < n ; col++ ) {
			/* Update (int,mom_1,mom_2,...) */
			for ( j = 0 ; j < ntilde ; j++ ) {
				moment[idxL][j] += filterPtr[col]*moment[idxG][j];
			}
			/* Jump to next Lambda coefficient */
			idxL += step2;
		}
		idxG += step2;   /* go to next Gamma coefficient and stay */
		/* in the same row of filter coefficients */
	}

	/* Cases where nbr. left Lambdas > nbr. right Lambdas */
	for ( row = top3 ; row >= 1 ; row-- ) {
		idxL = (top2-1)*step2;	// first Lambda is always in this place
		filterPtr -= n;   /* go to previous filter row */
		for ( col = n-1 ; col >= 0 ; col-- ) {
			/* Update (int,mom_1,mom_2,...) */
			for ( j = 0 ; j < ntilde ; j++ ) {
				moment[idxL][j] += filterPtr[col]*moment[idxG][j];
			}
			// Jump to next Lambda coefficient and next filter row
			idxL += step2;
		}
		idxG += step2;   /* go to next Gamma coefficient */
	}
}


//
// allocate a f.p. matrix with subscript range m[nrl..nrh][ncl..nch] 
//
template <class Data_T> double** Lifting1D<Data_T>::matrix(
	long nrl, long nrh, long ncl, long nch
) {
    long i;
    double	**m;

    /* allocate pointers to rows */
    m = new double* [nrh-nrl+1];
    if (!m) return NULL;
    m -= nrl;

    /* allocate rows and set pointers to them */
    for ( i=nrl ; i<=nrh ; i++ ) {
        m[i] = new double[nch-ncl+1];
        if (!m[i]) return NULL;
        m[i] -= ncl;
    }
    /* return pointer to array of pointers to rows */
    return m;
}

/* free a f.p. matrix llocated by matrix() */
template <class Data_T> void Lifting1D<Data_T>::free_matrix(
	double **m, long nrl, long nrh, long ncl, long nch 
) {
    long i;

	for ( i=nrh ; i>=nrl ; i-- ) {
		m[i] += ncl;
		delete [] m[i];
	}
    m += nrl;
    delete [] m;
}


/*
 *  -*- Mode: ANSI C -*-
 *  $Id: Lifting1D.h,v 1.8 2010/02/11 18:10:38 alannorton Exp $
 *  Author: Gabriel Fernandez
 *
 *  Definition of the functions used to perform the LU decomposition
 *  of matrix a. Function LUdcmp() performs the LU operations on a[][]
 *  and function LUbksb() performs the backsubstitution giving the
 *  solution in b[];
 */
/* do not edit anything above this line */


/* code */

/*
 * LUdcmp function: Given a matrix a [0..n-1][0..n-1], this routine
 *                  replaces it by the LU decomposition of a
 *                  rowwise permutation of itself. a and n are
 *                  input. a is output, arranged with L and U in
 *                  the same matrix; indx[0..n-1] is an output vector
 *                  that records the row permutation affected by
 *                  the partial pivoting; d is output as +/-1
 *                  depending on whether the number of row
 *                  interchanges was even or odd, respectively.
 *                  This routine is used in combination with LUbksb()
 *                  to solve linear equations or invert a matrix.
*/
template <class Data_T> void Lifting1D<Data_T>::LUdcmp(
	double **a, int n, int *indx, double *d
) {
    int i, imax, j, k;
    double big, dum, sum, temp;
    double *vv;   /* vv stores the implicit scaling of each row */

    vv=new double[n];
    *d=(double)1;               /* No row interchanges yet. */
    for (i=0;i<n;i++) {     /* Loop over rows to get the implicit scaling */
        big=(double)0;          /* information. */
        for (j=0;j<n;j++)
            if ((temp=(double)fabs((double)a[i][j])) > big)
                big=temp;
	assert(big != (double)0.0);	// singular matrix
        /* Nonzero largest element. */
        vv[i]=(double)1/big;    /* Save the scaling. */
    }
    for (j=0;j<n;j++) {     /* This is the loop over columns of Crout's method. */
        for (i=0;i<j;i++) {  /* Sum form of a triangular matrix except for i=j. */
            sum=a[i][j];     
            for (k=0;k<i;k++) sum -= a[i][k]*a[k][j];
            a[i][j]=sum;
        }
        big=(double)0;      /* Initialize for the search for largest pivot element. */
        imax = -1;        /* Set default value for imax */
        for (i=j;i<n;i++) {  /* This is i=j of previous sum and i=j+1...N */
            sum=a[i][j];      /* of the rest of the sum. */
            for (k=0;k<j;k++)
                sum -= a[i][k]*a[k][j];
            a[i][j]=sum;
            if ( (dum=vv[i]*(double)fabs((double)sum)) >= big) {
            /* Is the figure of merit for the pivot better than the best so far? */
                big=dum;
                imax=i;
            }
        }
        if (j != imax) {          /* Do we need to interchange rows? */
            for (k=0;k<n;k++) {  /* Yes, do so... */
                dum=a[imax][k];
                a[imax][k]=a[j][k];
                a[j][k]=dum;
            }
            *d = -(*d);           /* ...and change the parity of d. */
            vv[imax]=vv[j];       /* Also interchange the scale factor. */
        }
        indx[j]=imax;
        if (a[j][j] == (double)0.0)
            a[j][j]=(double)TINY;
        /* If the pivot element is zero the matrix is singular (at least */
        /* to the precision of the algorithm). For some applications on */
        /* singular matrices, it is desiderable to substitute TINY for zero. */
        if (j != n-1) {           /* Now, finally divide by pivot element. */
            dum=(double)1/(a[j][j]);
            for ( i=j+1 ; i<n ; i++ )
                a[i][j] *= dum;
        }
    }	/* Go back for the next column in the reduction. */
    delete [] vv;
}
    
    
/*
 *  LUbksb function: Solves the set of n linear equations A.X = B.
 *                   Here a[1..n][1..n] is input, not as the matrix A
 *                   but rather as its LU decomposition, determined
 *                   by the routine LUdcmp(). indx[1..n] is input as
 *                   the permutation vector returned by LUdcmp().
 *                   b[1..n] is input as the right hand side vector B,
 *                   and returns with the solution vector X. a, n, and
 *                   indx are not modified by this routine and can be
 *                   left in place for successive calls with different
 *                   right-hand sides b. This routine takes into account
 *                   the possibility that b will begin with many zero
 *                   elements, so it it efficient for use in matrix
 *                   inversion.
 */
template <class Data_T> void Lifting1D<Data_T>::LUbksb(
	double **a, int n, int *indx, double *b
) {
    int i,ii=-1,ip,j;
    double sum;

    for (i=0;i<n;i++) {   /* When ii is set to a positive value, it will */
        ip=indx[i];        /* become the index of the first nonvanishing */
        sum=b[ip];         /* element of b. We now do the forward substitution. */
        b[ip]=b[i];        /* The only new wrinkle is to unscramble the */
        if (ii>=0)            /* permutation as we go. */
            for (j=ii;j<=i-1;j++) sum -= a[i][j]*b[j];
        else if (sum)      /* A nonzero element was encountered, so from now on */
            ii=i;          /* we will have to do the sums in the loop above */
        b[i]=sum;
    }
    for (i=n-1;i>=0;i--) {   /* Now we do the backsubstitution. */
        sum=b[i];
        for (j=i+1;j<n;j++) sum -= a[i][j]*b[j];
        b[i]=sum/a[i][i];  /* Store a component of the solution vector X. */
        if (zero(b[i])) b[i] = (double)0;   /* Verify small numbers. */
    }                      /* All done! */
}


/*
 * UpdateLifting function: calculates the lifting coefficients using the given
 *                         integral-moment tuple.
 */
template <class Data_T> void Lifting1D<Data_T>::update_lifting(
	double *lifting,
	const double** moment,
	const int len,
	const int step2,
	const int noGammas,
	const int ntilde
) {
	int lcIdx,             /* index of lifting vector */
	i, j,              /* counters */
	row, col,          /* indices of the matrices */
	idxL, idxG,        /* pointers to Lambda & Gamma coeffs, resp. */
	top1, top2, top3;  /* number of iterations for L<ntilde, L=ntilde, L>ntilde */
	double** lift;	// used to find the lifting coefficients
	double *b;	// used to find the lifting coefficients
	int *indx;             	// used by the LU routines
	double d;	// used by the LU routines

	/**********************************/
	/* Calculate Lifting Coefficients */
	/**********************************/

	lcIdx = 0;         /* first element of lifting vector */

	/* Allocate space for the indx[] vector */
	indx = new int[ntilde];

	/* Allocate memory for the temporary lifting matrix and b */
	/* temporary matrix to be solved */
	lift = matrix(0, (long)ntilde-1, 0, (long)ntilde-1);   
	b = new double[ntilde];                  /* temporary solution vector */

	/* Calculate number of iterations for each case */
	top1 = (ntilde>>1) - 1;                 /* L < ntilde */
	top3 = (ntilde>>1) - IsOdd(len);          /* L > ntilde */
	top2 = noGammas - (top1 + top3);   /* L = ntilde */

	/* Cases where nbr. left Lambdas < nbr. right Lambdas */
	idxG = step2>>1;   /* second coefficient is Gamma */
	for ( i=0 ; i<top1 ; i++ ) {
		idxL = 0;   /* first Lambda is always the first coefficient */
		for (col=0 ; col<ntilde ; col++ ) {
			/* Load temporary matrix to be solved */
			for ( row=0 ; row<ntilde ; row++ ) {
				lift[row][col] = moment[idxL][row];	//matrix
			}
			/* Jump to next Lambda coefficient */
			idxL += step2;
		}
		/* Apply LU decomposition to lift[][] */
		LUdcmp (lift, ntilde, indx, &d);
		/* Load independent vector */
		for ( j=0 ; j<ntilde ; j++) {
			b[j] = moment[idxG][j];   /* independent vector */
		}
		/* Apply back substitution to find lifting coefficients */
		LUbksb (lift, ntilde, indx, b);
		for (col=0; col<ntilde; col++) {	// save them in lifting vector 
			lifting[lcIdx++] = b[col];
		}

		idxG += step2;   /* go to next Gamma coefficient */
	}

	/* Cases where nbr. left Lambdas = nbr. right Lambdas */
	for ( i=0 ; i<top2 ; i++ ) {
		idxL = i*step2;
		for ( col=0 ; col<ntilde ; col++ ) {
		/* Load temporary matrix to be solved */
			for ( row=0 ; row<ntilde ; row++ )
				lift[row][col] = moment[idxL][row];      /* matrix */
			/* Jump to next Lambda coefficient */
			idxL += step2;
		}
		/* Apply LU decomposition to lift[][] */
		LUdcmp (lift, ntilde, indx, &d);
		/* Load independent vector */
		for ( j=0 ; j<ntilde ; j++)
			b[j] = moment[idxG][j];   /* independent vector */
		/* Apply back substitution to find lifting coefficients */
		LUbksb (lift, ntilde, indx, b);
		for ( col=0 ; col<ntilde ; col++ )    /* save them in lifting vector */
			lifting[lcIdx++] = b[col];

		idxG += step2;   /* go to next Gamma coefficient */
	}

	/* Cases where nbr. left Lambdas > nbr. right Lambdas */
	for ( i=0 ; i<top3 ; i++ ) {
		idxL = (top2-1)*step2;   /* first Lambda is always in this place */
		for ( col=0 ; col<ntilde ; col++ ) {
			/* Load temporary matrix to be solved */
			for ( row=0 ; row<ntilde ; row++ )
				lift[row][col] = moment[idxL][row];      /* matrix */
			/* Jump to next Lambda coefficient */
			idxL += step2;
		}
		/* Apply LU decomposition to lift[][] */
		LUdcmp (lift, ntilde, indx, &d);
		/* Load independent vector */
		for ( j=0 ; j<ntilde ; j++)
			b[j] = moment[idxG][j];   /* independent vector */
		/* Apply back substitution to find lifting coefficients */
		LUbksb (lift, ntilde, indx, b);
		for (col=0;col<ntilde;col++) {	// save them in lifting vector
			lifting[lcIdx++] = b[col];
		}

	idxG += step2;   /* go to next Gamma coefficient */
	}

	/* Free memory */
	free_matrix(lift, 0, (long)ntilde-1, 0, (long)ntilde-1);
	delete [] indx;
	delete [] b;
}

/*
 * create_fwd_filter function: finds the filter coefficients used in the
 *                     Gamma coefficients prediction routine. The
 *                     Neville polynomial interpolation algorithm
 *                     is used to find all the possible values for
 *                     the filter coefficients. Thanks to this
 *                     process, the boundaries are correctly treated
 *                     without including artifacts.
 *                     Results are ordered in matrix as follows:
 *                         0 in the left and N   in the right
 *                         1 in the left and N-1 in the right
 *                         2 in the left and N-2 in the right
 *                                        .
 *                                        .
 *                                        .
 *                         N/2 in the left and N/2 in the right
 *                     For symmetry, the cases from
 *                         N/2+1 in the left and N/2-1
 *                     to
 *                         N in the left and 0 in the right
 *                     are the same as the ones shown above, but with
 *                     switched sign.
 */
template <class Data_T> double *Lifting1D<Data_T>::create_fwd_filter(
	unsigned int n
) {
	double xa[MAX_FILTER_COEFF], ya[MAX_FILTER_COEFF], x;
	int row, col, cc, Nrows;
	double *filter, *ptr;


	/* Number of cases for filter calculations */
	Nrows = (n>>1) + 1;    /* n/2 + 1 */

	/* Allocate memory for filter matrix */
	filter = new double[Nrows*n];
	if (! filter) return NULL;
	ptr = filter;

	/* Generate values of xa */
	xa[0] = (double)0.5*(double)(1 - (int) n);	// -n/2 + 0.5
	for (col = 1 ; col < (int)n ; col++) {
		xa[col] = xa[col-1] + (double)1;
	}

	/* Find filter coefficient values */
	filter += ( (Nrows*n) - 1 ); // go to last position in filter matrix
	for (row = 0 ; row < Nrows ; row++) {
		x = (double)row;
		for (col = 0 ; col < (int)n ; col++) {
			for (cc = 0 ; cc < (int)n ; cc++)
			ya[cc] = (double)0;
			ya[col] = (double)1;
			*(filter--) = neville (xa, ya, n, x);
		}
	}

	return ptr;
}

template <class Data_T> double *Lifting1D<Data_T>::create_inv_filter(
	unsigned int n, const double *filter
) {
	double	*inv_filter;
	int	Nrows;
	int	i;

	/* Number of cases for filter calculations */
	Nrows = (n>>1) + 1;    /* n/2 + 1 */

	/* Allocate memory for filter matrix */
	inv_filter = new double[Nrows*n];
	if (! inv_filter) return NULL;

        for ( i=0 ; i<(int)(n*(1+(n>>1))) ; i++ ) inv_filter[i] = -filter[i];

	return(inv_filter);
}

//
// create_moment function: Initializes the values of the Integral-Moments
//                      matrices. The moments are equal to k^i, where
//                      k is the coefficient index and i is the moment
//                      number (0,1,2,...).
//
template <class Data_T> double **Lifting1D<Data_T>::create_moment(
	const unsigned int ntilde, 
	const unsigned int width, 
	const int print
) {
	int row, col;

	double**	moment;

	/* Allocate memory for the Integral-Moment matrices */
	moment = matrix( 0, (long)(width-1), 0, (long)(ntilde-1) );

	/* Initialize Integral and Moments */
	/* Integral is equal to one at the beginning since all the filter */
	/* coefficients must add to one for each gamma coefficient. */
	/* 1st moment = k, 2nd moment = k^3, 3rd moment = k^3, ... */
	for ( row=0 ; row<(int)width ; row++ )    /* for the rows */
		for ( col=0 ; col<(int)ntilde ; col++ ) {
	if ( row==0 && col==0 )
		moment[row][col] = (double)1.0;   /* pow() domain error */
	else
		moment[row][col] = (double)pow( (double)row, (double)col );
	}

	/* Print Moment Coefficients */
	if (print) {
		fprintf (stdout, "Integral-Moments Coefficients for X (ntilde=%2d)\n", ntilde);
		for ( row=0 ; row<(int)width ; row++ ) {
			fprintf (stdout, "Coeff[%d] (%.20g", row, moment[row][0]);
			for ( col=1 ; col<(int)ntilde ; col++ )
				fprintf (stdout, ",%.20g", moment[row][col]);
			fprintf (stdout, ")\n");
		}
	}

	return moment;
}

#define CEIL(num,den) ( ((num)+(den)-1)/(den) )

template <class Data_T> double *Lifting1D<Data_T>::create_inv_lifting(
	double	*fwd_lifting,
	unsigned int	ntilde,
	unsigned int	width
) {
	int	noGammas;
	double	*inv_lifting = NULL;
	int	col;

	/* Allocate lifting coefficient matrices */
	noGammas = width >> 1;    	// number of Gammas 
	if (noGammas > 0) {
		inv_lifting = new double[noGammas*ntilde];
		for ( col=0 ; col<(int)(noGammas*ntilde) ; col++ ) {
			inv_lifting[col] = -fwd_lifting[col];
		}
	}
	return(inv_lifting);
}

//
// create_fwd_lifting : Updates corresponding moment matrix (X if dir is FALSE
//                      and Y if dir is TRUE) and calculates the lifting
//                      coefficients using the new moments.
//                      This function is used for the forward transform and
//                      uses the original filter coefficients.
//
template <class Data_T> double *Lifting1D<Data_T>::create_fwd_lifting(
	const double *filter, 
	const unsigned int n, 
	const unsigned int ntilde, 
	const unsigned int width
) {
	double** moment;   /* pointer to the Integral-Moments matrix */
	int len;         /* number of coefficients in this level */
	int step2;      /* step size between same coefficients */
	int noGammas;    /* number of Gamma coeffs */
	double *lifting = NULL;

	int col;

	moment = create_moment(ntilde, width, 0);
	if (! moment) return NULL;


	/**********************************/
	/* Initialize important constants */
	/**********************************/

	/* Calculate number of coefficients and step size */
	len = CEIL (width, 1);
	step2 = 1 << 1;

	/* Allocate lifting coefficient matrices */
	noGammas = width >> 1;    	// number of Gammas 
	if (noGammas > 0) {
		lifting = new double[noGammas*ntilde];
		for ( col=0 ; col<(int)(noGammas*ntilde) ; col++ ) {
			lifting[col] = (double)0;
		}
	}

	update_moment(moment, filter, step2, noGammas, len, n, ntilde );
	update_lifting(
		lifting, (const double **) moment, len, step2, 
		noGammas, ntilde
	);

#ifdef	DEBUGING
	fprintf (stdout, "\nLifting Coefficients:\n");
	liftPtr = lifting;
	for ( x=0 ; x<(width>>1) ; x++ ) {
		sprintf (buf, "%.20g", *(liftPtr++));
		for ( k=1 ; k<nTilde ; k++ )
			sprintf (buf, "%s,%.20g", buf, *(liftPtr++));
		fprintf (stdout, "Gamma[%d] (%s)\n", x, buf);
	}
#endif

	/* Free memory allocated for the moment matrices */
	free_matrix(moment, 0, (long)(width-1), 0, (long)(ntilde-1) );

	return(lifting);
}




/*
 * FLWT1D_Predict function: The Gamma coefficients are found as an average
 *                          interpolation of their neighbors in order to find
 *                          the "failure to be linear" or "failure to be
 *                          cubic" or the failure to be the order given by
 *                          N-1. This process uses the filter coefficients
 *                          stored in the filter vector and predicts
 *                          the odd samples based on the even ones
 *                          storing the difference in the gammas.
 *                          By doing so, a Dual Wavelet is created.
 */
template <class Data_T> void Lifting1D<Data_T>::FLWT1D_Predict(
	Data_T* vect,
	const long width,
	const long N,
	double *filter, 
	int stride
) {
    register Data_T *lambdaPtr,	//pointer to Lambda coeffs
			*gammaPtr;		// pointer to Gamma coeffs
    register double *fP, *filterPtr;   	// pointers to filter coeffs
    register long len,              	// number of coeffs at current level
                  j,                 /* counter for the filter cases */
                  stepIncr;          /* step size between coefficients of the same type */

    long stop1,                      /* number of cases when L < R */
         stop2,                      /* number of cases when L = R */
         stop3,                      /* number of cases when L > R */
         soi;                        /* increment for the middle cases */

    /************************************************/
    /* Calculate values of some important variables */
    /************************************************/
    len       = CEIL(width, 1);   /* number of coefficients at current level */
    stepIncr  = stride << 1;   /* step size betweeen coefficients */

    /************************************************/
    /* Calculate number of iterations for each case */
    /************************************************/
    j     = IsOdd(len);
    stop1 = N >> 1;
    stop3 = stop1 - j;                /* L > R */
    stop2 = (len >> 1) - N + 1 + j;   /* L = R */
    stop1--;                          /* L < R */

    /***************************************************/
    /* Predict Gamma (wavelet) coefficients (odd guys) */
    /***************************************************/

    filterPtr = filter + N;   /* position filter pointer */

    /* Cases where nbr. left Lambdas < nbr. right Lambdas */
    gammaPtr = vect + (stepIncr >> 1);   /* second coefficient is Gamma */
    while(stop1--) {
        lambdaPtr = vect;   /* first coefficient is always first Lambda */
        j = N;
        do {   /* Gamma update (Gamma - predicted value) */
            *(gammaPtr) -= (*(lambdaPtr)*(*(filterPtr++)));
            lambdaPtr   += stepIncr;   /* jump to next Lambda coefficient */
        } while(--j);   /* use all N filter coefficients */
        /* Go to next Gamma coefficient */
        gammaPtr += stepIncr;
    }

    /* Cases where nbr. left Lambdas = nbr. right Lambdas */
    soi = 0;
    while(stop2--) {
        lambdaPtr = vect + soi;   /* first Lambda to be read */
        fP = filterPtr;   /* filter stays in same values for this cases */
        j = N;
        do {   /* Gamma update (Gamma - predicted value) */
            *(gammaPtr) -= (*(lambdaPtr)*(*(fP++)));
            lambdaPtr   += stepIncr;   /* jump to next Lambda coefficient */
        } while(--j);   /* use all N filter coefficients */
        /* Move start point for the Lambdas */
        soi += stepIncr;
        /* Go to next Gamma coefficient */
        gammaPtr += stepIncr;
    }

    /* Cases where nbr. left Lambdas > nbr. right Lambdas */
    fP = filterPtr;   /* start going backwards with the filter coefficients */
    vect += (soi-stepIncr);   /* first Lambda is always in this place */
    while (stop3--) {
        lambdaPtr = vect;   /* position Lambda pointer */
        j = N;
        do {   /* Gamma update (Gamma - predicted value) */
            *(gammaPtr) -= (*(lambdaPtr)*(*(--fP)));
            lambdaPtr   += stepIncr;   /* jump to next Lambda coefficient */
        } while(--j);   /* use all N filter coefficients */
        /* Go to next Gamma coefficient */
        gammaPtr += stepIncr;
    }
}

/*
 * FLWT1D_Update: the Lambda coefficients have to be "lifted" in order
 *                to find the real wavelet coefficients. The new Lambdas
 *                are obtained  by applying the lifting coeffients stored
 *                in the lifting vector together with the gammas found in
 *                the prediction stage. This process assures that the
 *                moments of the wavelet function are preserved.
 */
template <class Data_T> void Lifting1D<Data_T>::FLWT1D_Update(
	Data_T* vect,
	const long width,
	const long nTilde,
	const double *lc,
	int stride
) {
    const register double * lcPtr;   /* pointer to lifting coefficient values */
    register Data_T	*vL,	/* pointer to Lambda values */
					*vG;	/* pointer to Gamma values */
    register long j;         /* counter for the lifting cases */

    long len,                /* number of coefficietns at current level */
         stop1,              /* number of cases when L < R */
         stop2,              /* number of cases when L = R */
         stop3,              /* number of cases when L > R */
         noGammas,           /* number of Gamma coefficients at this level */
         stepIncr,           /* step size between coefficients of the same type */
         soi;                /* increment for the middle cases */


    /************************************************/
    /* Calculate values of some important variables */
    /************************************************/
    len      = CEIL(width, 1);   /* number of coefficients at current level */
    stepIncr = stride << 1;   /* step size between coefficients */
    noGammas = len >> 1 ;          /* number of Gamma coefficients */

    /************************************************/
    /* Calculate number of iterations for each case */
    /************************************************/
    j	  = IsOdd(len);
    stop1 = nTilde >> 1;
    stop3 = stop1 - j;                   /* L > R */
    stop2 = noGammas - nTilde + 1 + j;   /* L = R */
    stop1--;                             /* L < R */

    /**********************************/
    /* Lift Lambda values (even guys) */
    /**********************************/

    lcPtr = lc;   /* position lifting pointer */

    /* Cases where nbr. left Lambdas < nbr. right Lambdas */
    vG = vect + (stepIncr >> 1);   /* second coefficient is Gamma */
    while(stop1--) {
        vL = vect;   /* lambda starts always in first coefficient */
        j = nTilde;
        do {
            *(vL) += (*(vG)*(*(lcPtr++)));   /* lift Lambda (Lambda + lifting value) */
            vL    += stepIncr;               /* jump to next Lambda coefficient */
        } while(--j);   /* use all nTilde lifting coefficients */
        /* Go to next Gamma coefficient */
        vG += stepIncr;
    }

    /* Cases where nbr. left Lambdas = nbr. right Lambdas */
    soi = 0;
    while(stop2--) {
        vL = vect + soi;   /* first Lambda to be read */
        j = nTilde;
        do {
            *(vL) += (*(vG)*(*(lcPtr++)));   /* lift Lambda (Lambda + lifting value) */
            vL    += stepIncr;               /* jump to next Lambda coefficient */
        } while(--j);   /* use all nTilde lifting coefficients */
        /* Go to next Gamma coefficient */
        vG  += stepIncr;
        /* Move start point for the Lambdas */
        soi += stepIncr;
    }

    /* Cases where nbr. left Lambdas = nbr. right Lambdas */
    vect += (soi - stepIncr);   /* first Lambda is always in this place */
    while(stop3--) {
        vL = vect;   /* position Lambda pointer */
        j = nTilde;
        do {
            *(vL) += (*(vG)*(*(lcPtr++)));   /* lift Lambda (Lambda + lifting value) */
            vL    += stepIncr;               /* jump to next Lambda coefficient */
        } while(--j);   /* use all nTilde lifting coefficients */
        /* Go to next Gamma coefficient */
        vG += stepIncr;
    }
}

template <class Data_T> void Lifting1D<Data_T>::forward_transform1d_haar(
	Data_T *data,
	int width,
	int stride
) {
	int	i;

	int	nG;	// # gamma coefficients
	int	nL;	// # lambda coefficients
	double	lsum = 0.0;	// sum of lambda values
	double	lave = 0.0;	// average of lambda values
	int stepIncr = stride << 1;   // step size betweeen coefficients


    nG = (width >> 1);
    nL = width - nG;

	//
	// Need to preserve average for odd sizes
	//
	if (IsOdd(width)) {
		double	t = 0.0;

		for(i=0;i<width;i++) {
			t += data[i*stride];
		}
		lave = t / (double) width;
	}

	for (i=0; i<nG; i++) {
		data[stride] = data[stride] - data[0];	// gamma
		data[0] = (Data_T)(data[0] + (data[stride] /2.0)); // lambda
		lsum += data[0];

		data += stepIncr;
	}

    // If IsOdd(width), then we have one additional case for */
    // the Lambda calculations. This is a boundary case  */
    // and, therefore, has different filter values.      */
	//
	if (IsOdd(width)) {
		data[0] = (Data_T)((lave * (double) nL) - lsum);
	}
}



template <class Data_T> void Lifting1D<Data_T>::inverse_transform1d_haar(
	Data_T *data,
	int width,
	int stride
) {
	int	i,j;
	int	nG;	// # gamma coefficients
	int	nL;	// # lambda coefficients
	double	lsum = 0.0;	// sum of lambda values
	double	lave = 0.0;	// average of lambda values

	int stepIncr = stride << 1;   // step size betweeen coefficients

	nG = (width >> 1);
	nL = width - nG;

    // Odd # of coefficients require special handling at boundary
	// Calculate Lambda average 
	//
    if (IsOdd(width) ) {
        double  t = 0.0;

		for(i=0,j=0;i<nL;i++,j+=stepIncr) {
            t += data[j];
        }
        lave = t/(double)nL;   // average we've to maintain
    }

	for (i=0; i<nG; i++) {
		data[0] = (Data_T)(data[0] - (data[stride] * 0.5));
		data[stride] = data[stride] + data[0];
		lsum += data[0] +  data[stride];

		data += stepIncr;
	}

    // If ODD(len), then we have one additional case for */
    // the Lambda calculations. This is a boundary case  */
    // and, therefore, has different filter values.      */
	//
    if (IsOdd(width)) {
        *data = (Data_T)((lave * (double) width) - lsum);
    }
}

#endif	//	_Lifting1D_h_
