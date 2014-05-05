
#ifndef	_MatWaveDwt_h_
#define	_MatWaveDwt_h_

#include "MatWaveBase.h"

namespace VAPoR {

//
//! \class MatWaveDwt
//! \brief Implements a single level wavelet filter
//! \author John Clyne
//! \version $Revision: 1.1 $
//! \date    $Date: 2010/06/07 16:43:10 $
//!
//! The MatWaveDwt class provides single-level wavelet filters similar
//! to those provided by the Matlab dwt and idwt functions. 1D, 2D, and 3D
//! transforms are provided. The API for dwt and idwt more closely 
//! matches the MatLab wavedec and waverec functions than the MatLab
//! functions of the same name.
//!
class MatWaveDwt : public MatWaveBase {

public:

 //! Create a wavelet filter bank 
 //!
 //! \param[in] wname The name of the wavelet to apply.
 //! \param[in] mode The boundary extension mode.
 //!
 //! \sa dwtmode()
 //!
 MatWaveDwt(const string &wname, const string &mode);
 virtual ~MatWaveDwt();

 //! Single-level discrete 1D wavelet transform
 //!
 //! This method performs a single-level, one-dimensional wavelet 
 //! decomposition with respect to the current wavelet 
 //! and boundary extension mode.
 //!
 //! \param[in] sigIn The discrete signal
 //! \param[in] sigInLength The length of \p sigIn
 //! \param[out] C The wavelet decompostion vector. The length of \p C, 
 //! must be equal to
 //! the value returned by MatWaveWavedec::coefflength().
 //! \param[out] cA The wavelet decompostion vector approximation coefficients
 //! \param[out] cD The wavelet decompostion vector detail coefficients
 //! \param[out] L[3] The book keeping vector.  The length of \p L, must 
 //! be equal to 3. \p L[0] provides the length of the approximation
 //! coefficients, \p L[1] provides the length of the detail coefficients,
 //! and \p L[2] is equal to \p sigInLength.
 //!
 //! \retval status A negative number indicates failure.
 //!
 //! \sa MatWaveBase::coefflength(), idwt()
 //
 int dwt(const double *sigIn, size_t sigInLength, double *C, size_t L[3]);
 int dwt(const float *sigIn, size_t sigInLength, float *C, size_t L[3]);
 int dwt(const double *sigIn, size_t sigInLength, double *cA, double *cD, size_t L[3]);
 int dwt(const float *sigIn, size_t sigInLength, float *cA, float *cD, size_t L[3]);

 //! Single-level inverse discrete 1D wavelet transform
 //!
 //! This method performs a single-level, one-dimensional wavelet 
 //! reconstruction with respect to the current wavelet and
 //! boundary extension mode.
 //!
 //! \param[in] C The Wavelet decomposition vector, dimensioned according
 //! to \p L.
 //! \param[in] cA The wavelet decompostion vector approximation coefficients
 //! \param[in] cD The wavelet decompostion vector detail coefficients
 //! \param[in] L[3] The Wavelet decomposition book keeping vector.
 //! \param[out] sigOut Single-level reconstruction approximation based
 //! on the approximation and detail coefficients (\p C). The length of
 //! \p sigOut is must be \p L[2].
 //!
 //! \retval status A negative number indicates failure.
 //!
 //! \sa MatWaveBase::coefflength(), dwt()
 //
 int idwt(const double *C, const size_t L[3], double *sigOut); 
 int idwt(const float *C, const size_t L[3], float *sigOut); 
 int idwt(const double *cA, const double *cD, const size_t L[3], double *sigOut); 
 int idwt(const float *cA, const float *cD, const size_t L[3], float *sigOut); 

 //! Single-level discrete 2D wavelet transform
 //!
 //! This method performs a single-level, two-dimensional wavelet 
 //! decomposition with respect to the current wavelet and 
 //! boundary extension mode.
 //!
 //! \param[in] sigIn The discrete signal
 //! \param[in] sigInX The length of the X dimension of \p sigIn
 //! \param[in] sigInY The length of the Y dimension of \p sigIn
 //! \param[out] C The wavelet decompostion vector. The length of \p C, 
 //! must be equal to
 //! the value returned by MatWaveWavedec::coefflength2().
 //! \param[out] cA The wavelet decompostion vector approximation coefficients
 //! \param[out] cDh The wavelet decompostion vector horizontal 
 //! detail coefficients
 //! \param[out] cDv The wavelet decompostion vector vertical 
 //! detail coefficients
 //! \param[out] cDv The wavelet decompostion vector diagonal 
 //! detail coefficients
 //! \param[out] L[10] The book keeping vector.  The length of \p L, must 
 //! be equal to 6 + 4. \p L[0] and \L[1]  provide the dimensions of 
 //! the approximation
 //! coefficients, \p L[2] and \p L[3] provides the dimension of the 
 //! horizontal detail coefficients, \p L[4] and \p L[5] the horizontal
 //! coefficients, \p L[6] and \p L[7] the diagonal detail coefficients,
 //! and \p L[8] \p L[9] are equal to \p sigInX and \p sigInY, respectively.
 //!
 //!
 //! \retval status A negative number indicates failure.
 //!
 //! \sa MatWaveBase::coefflength(), idwt()
 //
 int dwt2d(
	const double *sigIn, size_t sigInX, size_t sigInY, double *C, size_t L[10]
 );
 int dwt2d(
	const float *sigIn, size_t sigInX, size_t sigInY, float *C, size_t L[10]
 );
 int dwt2d(
	const double *sigIn, size_t sigInX, size_t sigInY,
	double *cA, double *cDh, double *cDv, double *cDd, size_t L[10]
 );
 int dwt2d(
	const float *sigIn, size_t sigInX, size_t sigInY,
	float *cA, float *cDh, float *cDv, float *cDd, size_t L[10]
 );

 //! Single-level inverse discrete 2D wavelet transform
 //!
 //! This method performs a single-level, two-dimensional wavelet 
 //! reconstruction with respect to the current wavelet and boundary
 //! extension mode.
 //!
 //! \param[in] C The Wavelet decomposition vector, dimensioned according
 //! to \p L.
 //! \param[in] cA The wavelet decompostion vector approximation coefficients
 //! \param[in] cDh The wavelet decompostion vector horizontal 
 //! detail coefficients
 //! \param[in] cDv The wavelet decompostion vector vertical 
 //! detail coefficients
 //! \param[in] cDv The wavelet decompostion vector diagonal 
 //! detail coefficients
 //! \param[in] L[10] The Wavelet decomposition book keeping vector.
 //! \param[out] sigOut Single-level reconstruction approximation based
 //! on the approximation and detail coefficients (\p C). The length of
 //! \p sigOut is must be \p L[8] * \p L[9].
 //!
 //! \retval status A negative number indicates failure.
 //!
 //! \sa MatWaveBase::coefflength(), dwt()
 //
 int idwt2d(const double *C, const size_t L[10], double *sigOut);
 int idwt2d(const float *C, const size_t L[10], float *sigOut);
 int idwt2d(
	const double *cA, const double *cDh, const double *cDv, const double *cDd,
	const size_t L[10], double *sigOut
 );
 int idwt2d(
	const float *cA, const float *cDh, const float *cDv, const float *cDd,
	const size_t L[10], float *sigOut
 );

 //! Single-level discrete 3D wavelet transform
 //!
 //! C is partitioned in the order: LLL, LLH, LHL, LHH, HLL,
 //! HLH, HHL, HHH
 //!
 int dwt3d(
	const double *sigIn, size_t sigInX, size_t sigInY, size_t sigInZ, 
	double *C, size_t L[27]
 );
 int dwt3d(
	const float *sigIn, size_t sigInX, size_t sigInY, size_t sigInZ, 
	float *C, size_t L[27]
 );

 //! Single-level inverse discrete 3D wavelet transform
 //
 int idwt3d(const double *C, const size_t L[27], double *sigOut);
 int idwt3d(const float *C, const size_t L[27], float *sigOut);
 int idwt3d(
    const double *cLLL, const double *cLLH, const double *cLHL, 
	const double *cLHH,
    const double *cHLL, const double *cHLH, const double *cHHL, 
	const double *cHHH,
    const size_t L[27], double *sigOut
 );
 int idwt3d(
    const float *cLLL, const float *cLLH, const float *cLHL, const float *cLHH,
    const float *cHLL, const float *cHLH, const float *cHHL, const float *cHHH,
    const size_t L[27], float *sigOut
 );

private:

 // 1D buffers
 size_t _dwt1dBufSize;
 double *_dwt1dBuf;

 // 2D buffers
 size_t _dwt2dBufSize;
 double *_dwt2dBuf;

 // 3D buffers
 size_t _dwt3dBuf1Size;
 double *_dwt3dBuf1;
 size_t _dwt3dBuf2Size;
 double *_dwt3dBuf2;

};

}

#endif


