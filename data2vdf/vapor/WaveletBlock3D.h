//
//      $Id: WaveletBlock3D.h,v 1.7 2009/12/10 19:21:02 clynejp Exp $
//

#ifndef	_WaveletBlock3D_h_
#define	_WaveletBlock3D_h_

#include <vapor/MyBase.h>
#include <vapor/EasyThreads.h>
#include <vapor/WaveletBlock1D.h>

#include "Lifting1D.h"


namespace VAPoR {

//
//! \class WaveletBlock3D
//! \brief A block-based, 3D wavelet transformer
//! \author John Clyne
//! \version $Revision: 1.7 $
//! \date    $Date: 2009/12/10 19:21:02 $
//!
//! This class provides a 3D, block-based wavelet transform API
//! based on Wim Swelden's Liftpack library.
//
class WaveletBlock3D : public VetsUtil::MyBase {

public:

 //! Constructor for the WaveletBlock3D class.
 //! \param[in] bs Dimension of a block along X, Y, and Z coordinates
 //! axis
 //! \param[in] Number of wavelet filter coefficients. Valid values are
 //! from 1 to Lifting1D::MAX_FILTER_COEFF
 //! \param[in] ntilde Number of wavelet lifting coefficients. Valid values are
 //! from 1 to Lifting1D::MAX_FILTER_COEFF
 //! \param[in] nthreads Number of execution threads that may be used by
 //! the class for parallel execution.
 //! \note The success or failure of this constructor can be checked
 //! with the GetErrCode() method.
 //!
 //! \sa Lifting1D, GetErrCode(),
 //
 WaveletBlock3D(
	unsigned int bs,	// X,Y,Z coordinate block dimensions
	unsigned int n,		// # wavelet filter coefficents
	unsigned int ntilde,	// # wavelet lifting coefficients
	unsigned int nthreads
 );
 virtual ~WaveletBlock3D();

 //! Perform a forward wavelet transform on a super block
 //!
 //! Performs a forward, 3D wavelet transform on each of eight neighboring
 //! blocks contained in a super block, \p src_super_blk.
 //! The resulting Lambda and Gamma coefficients are stored in a
 //! destination super block, \p dst_super_blk. 
 //! The distribution of coefficients stored in the space pointed to by
 //! dst_super_blk is as follows: block 0 contains the L (Lambda) 
 //! subband, block 1 contains the Hz (Gamma) subband, blocks 2 & 3 
 //! contain the Hy subband, blocks 4 - 7 contain the Hx subband.
 //!
 //! \param[in] src_super_blk An array of pointers to blocks. 
 //! \param[out] dst_super_blk An array of pointers to blocks. 
 //!
 //! \note The resulting transformed coefficients are stored as tranposed
 //! arrays.  The blocks may be re-ordered to conventional, X, Y, Z order
 //! with the \p Transpose() method.
 //! 
 //! \sa InverseTransform(), Transpose()
 //
 void	ForwardTransform(
	const float *src_super_blk[8],
	float *dst_super_blk[8]
 );

 //! Perform an inverse wavelet transform on a super block
 //!
 //! Performs a inverse, 3D wavelet transform on each of eight neighboring
 //! blocks contained in a super block, \p src_super_blk.
 //! The resulting coefficients are stored in a
 //! destination super block, \p dst_super_blk. 
 //!
 //! \param[in] src_super_blk An array of pointers to blocks  
 //! \param[out] dst_super_blk An array of pointers to blocks. 
 //!
 //! \sa ForwardTransform()
 //
 void	InverseTransform(
	const float *src_super_blk[8],
	float *dst_super_blk[8]
 );

 //! Transpose a super block created by ForwardTransform() to X, Y, Z order.
 //!
 //! The gamma coefficient blocks returned by ForwardTransform() are 
 //! tranposed. This method can be used to perform an in-place tranpose
 //! restoring gamma blocks to their natural, X, Y, Z order.
 //! \param[in,out] src_super_blk An array of pointers to blocks  
 //!
 //! \sa ForwardTransform()
 //
 void TransposeBlks(
	float *super_blk[8]
 );

 void	inverse_transform_thread();

private:
 int	_objInitialized;	// has the obj successfully been initialized?
 const float **src_super_blk_c;
 float **dst_super_blk_c;
 const float ***src_s_blk_ptr_c;
 float ***dst_s_blk_ptr_c;
 int	bs_c;			// block dimensions in voxels
 int	nthreads_c;		// # execution threads
 float	**temp_blks1_c,
	**temp_blks2_c;

 WaveletBlock1D *_wb1d;	// lifting method wavelet transform

 int	deallocate_c;		// execute destructor for this object?

 VetsUtil::EasyThreads	*et_c;

 WaveletBlock3D	**threads_c;	// worker threads

 int	z0_c, zr_c;	// decomposition work boundaries for threads

 WaveletBlock3D(WaveletBlock3D *X, int index);

 void	forward_transform3d_blocks(
	const float **src_blks,
	float **lambda_blks,
	float **gamma_blks,
	int nblocks
 );

 void	forward_transform3d(
	const float *src_blkptr,
	float *lambda_blkptr,
	float *gamma_blkptr,
	int lambda_offset,
	int gamma_offset
 );

 void	inverse_transform3d_blocks(
	const float **lambda_blks,
	const float **gamma_blks,
	float **dst_blks,
	int nblocks
 );

 void	inverse_transform3d(
	const float *lambda_blkptr,
	const float *gamma_blkptr,
	float *dst_blkptr,
	int lambda_offset,
	int gamma_offset
 );

};

};

#endif	//	_WaveletBlock3D_h_
