//
//      $Id: WaveletBlock2D.h,v 1.3 2009/12/10 19:21:02 clynejp Exp $
//

#ifndef	_WaveletBlock2D_h_
#define	_WaveletBlock2D_h_

#include <vapor/MyBase.h>
#include <vapor/EasyThreads.h>
#include <vapor/WaveletBlock1D.h>

#include "Lifting1D.h"


namespace VAPoR {

//
//! \class WaveletBlock2D
//! \brief A tile-based, 2D wavelet transformer
//! \author John Clyne
//! \version $Revision: 1.3 $
//! \date    $Date: 2009/12/10 19:21:02 $
//!
//! This class provides a 2D, tile-based wavelet transform API
//! based on Wim Swelden's Liftpack library.
//
class WaveletBlock2D : public VetsUtil::MyBase {

public:

 //! Constructor for the WaveletBlock2D class.
 //! \param[in] bs Dimension of a tile along X, Y, and Z coordinates
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
 WaveletBlock2D(
	size_t bs[2],	// X,Y,Z coordinate tile dimensions
	unsigned int n,		// # wavelet filter coefficents
	unsigned int ntilde	// # wavelet lifting coefficients
 );
 virtual ~WaveletBlock2D();

 //! Perform a forward wavelet transform on a super tile
 //!
 //! Performs a forward, 2D wavelet transform on each of eight neighboring
 //! tiles contained in a super tile, \p src_super_tile.
 //! The resulting Lambda and Gamma coefficients are stored in a
 //! destination super tile, \p dst_super_tile. 
 //! The distribution of coefficients stored in the space pointed to by
 //! dst_super_tile is as follows: tile 0 contains the L (Lambda) 
 //! subband, tile 1 contains the Hz (Gamma) subband, tiles 2 & 3 
 //! contain the Hy subband, tiles 4 - 7 contain the Hx subband.
 //!
 //! \param[in] src_super_tile An array of pointers to tiles. 
 //! \param[out] dst_super_tile An array of pointers to tiles. 
 //!
 //! \note The resulting transformed coefficients are stored as tranposed
 //! arrays.  The tiles may be re-ordered to conventional, X, Y, Z order
 //! with the \p Transpose() method.
 //! 
 //! \sa InverseTransform(), Transpose()
 //
 void	ForwardTransform(
	const float *src_super_tile[4],
	float *dst_super_tile[4]
 );

 //! Perform an inverse wavelet transform on a super tile
 //!
 //! Performs a inverse, 2D wavelet transform on each of eight neighboring
 //! tiles contained in a super tile, \p src_super_tile.
 //! The resulting coefficients are stored in a
 //! destination super tile, \p dst_super_tile. 
 //!
 //! \param[in] src_super_tile An array of pointers to tiles  
 //! \param[out] dst_super_tile An array of pointers to tiles. 
 //!
 //! \sa ForwardTransform()
 //
 void	InverseTransform(
	const float *src_super_tile[4],
	float *dst_super_tile[4]
 );

private:
 int	_objInitialized;	// has the obj successfully been initialized?
 size_t	_bs[2];			// tile dimensions in voxels
 float	**_temp_tiles1,
	**_temp_tiles2;

 WaveletBlock1D *_wb1d0;	// lifting method wavelet transform
 WaveletBlock1D *_wb1d1;	// lifting method wavelet transform

 WaveletBlock2D(WaveletBlock2D *X, int index);

 void	forward_transform2D_tiles(
	const float **src_tiles,
	float **lambda_tiles,
	float **gamma_tiles,
	int ntiles,
	size_t nx,
	size_t ny,
	WaveletBlock1D *wb1d
 );

 void	forward_transform2D(
	const float *src_tileptr,
	float *lambda_tileptr,
	float *gamma_tileptr,
	int lambda_offset,
	int gamma_offset,
	size_t nx,
	size_t ny,
	WaveletBlock1D *wb1d
 );

 void	inverse_transform2D_tiles(
	const float **lambda_tiles,
	const float **gamma_tiles,
	float **dst_tiles,
	int ntiles,
	size_t nx,
	size_t ny,
	WaveletBlock1D *wb1d
 );

 void	inverse_transform2D(
	const float *lambda_tileptr,
	const float *gamma_tileptr,
	float *dst_tileptr,
	int lambda_offset,
	int gamma_offset,
	size_t nx,
	size_t ny,
	WaveletBlock1D *wb1d
 );

};

};

#endif	//	_WaveletBlock2D_h_
