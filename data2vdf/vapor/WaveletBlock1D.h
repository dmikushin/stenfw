//
//      $Id: WaveletBlock1D.h,v 1.3 2009/12/10 19:21:02 clynejp Exp $
//

#ifndef	_WaveletBlock1D_h_
#define	_WaveletBlock1D_h_

#include <vapor/MyBase.h>
#include <vapor/EasyThreads.h>

#include "Lifting1D.h"


namespace VAPoR {

//
//! \class WaveletBlock1D
//! \brief A block-based, 1D wavelet transformer
//! \author John Clyne
//! \version $Revision: 1.3 $
//! \date    $Date: 2009/12/10 19:21:02 $
//!
//! This class provides a 1D, block-based wavelet transform API
//! based on Wim Swelden's Liftpack library.
//
class WaveletBlock1D : public VetsUtil::MyBase {

public:

 //! Constructor for the WaveletBlock1D class.
 //! \param[in] bs Block length
 //! \param[in] Number of wavelet filter coefficients. Valid values are
 //! from 1 to Lifting1D::MAX_FILTER_COEFF
 //! \param[in] ntilde Number of wavelet lifting coefficients. Valid values are
 //! from 1 to Lifting1D::MAX_FILTER_COEFF
 //! \note The success or failure of this constructor can be checked
 //! with the GetErrCode() method.
 //!
 //! \sa Lifting1D, GetErrCode(),
 //
 WaveletBlock1D(
	unsigned int bs,	// X,Y,Z coordinate block dimensions
	unsigned int n,		// # wavelet filter coefficents
	unsigned int ntilde	// # wavelet lifting coefficients
 );
 virtual ~WaveletBlock1D();

 void	ForwardTransform(
	const float *src_blk_ptr,
	float *lambda_blk_ptr,
	float *gamma_blk_ptr
 );
 void	InverseTransform(
	const float *lambda_blk_ptr,
	const float *gamma_blk_ptr,
	float *dst_blk_ptr
 );

private:
 int	_bs;			// block dimensions in voxels
 int	_n;				// # filter coefficients
 int	_ntilde;		// # lifting coefficients
 int	_nthreads;		// # execution threads

 Lifting1D <float>	*_lift;	// lifting method wavelet transform
 float		*_liftbuf;	// scratch space for lifting method



 void	forward_transform1d_haar(
	const float *src_blk_ptr,
	float *lambda_blk_ptr,
	float *gamma_blk_ptr,
	int size
 );


 void	inverse_transform1d_haar(
	const float *lambda_blk_ptr,
	const float *gamma_blk_ptr,
	float *src_blk_ptr,
	int size
 );

};

};

#endif	//	_WaveletBlock1D_h_
