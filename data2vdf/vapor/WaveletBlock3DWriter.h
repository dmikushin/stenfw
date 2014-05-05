//
//      $Id: WaveletBlock3DWriter.h,v 1.12 2010/09/24 17:14:07 southwic Exp $
//

#ifndef	_WavletBlock3DWriter_h_
#define	_WavletBlock3DWriter_h_

#include <vapor/MyBase.h>
#include <vapor/WaveletBlockIOBase.h>

namespace VAPoR {

//
//! \class WaveletBlock3DWriter
//! \brief A slab writer for VDF files
//! \author John Clyne
//! \version $Revision: 1.12 $
//! \date    $Date: 2010/09/24 17:14:07 $
//!
//! This class provides a low-level API for writing data volumes to
//! a VDF file. The Write methods contained within are the  most efficient
//! (both in terms of memory and performance) for writing an entire data
//! volume.
//
class VDF_API	WaveletBlock3DWriter : public WaveletBlockIOBase {

public:

 //! Constructor for the WaveletBlock3DWriter class.
 //! \param[in,out] metadata A pointer to a Metadata structure identifying the
 //! data set upon which all future operations will apply.
 //! \note The success or failure of this constructor can be checked
 //! with the GetErrCode() method.
 //!
 //! \sa MetadataVDC, GetErrCode()
 //
 WaveletBlock3DWriter(
	const MetadataVDC &metadata
 );

 //! Constructor for the WaveletBlock3DWriter class.
 //! \param[in] metafile Path to a metadata file for which all
 //! future class operations will apply
 //! \note The success or failure of this constructor can be checked
 //! with the GetErrCode() method.
 //!
 //! \sa MetadataVDC, GetErrCode()
 //
 WaveletBlock3DWriter(
	const string &metafile
 );

 virtual ~WaveletBlock3DWriter();

 //! Open the named variable for writing
 //!
 //! Prepare a vapor data file for the creation of a multiresolution
 //! data volume via subsequent write operations by
 //! other methods of this classes derived from this class.
 //! The data volume is identified by the specfied time step and
 //! variable name. The number of forward transforms applied to
 //! the volume is determined by the Metadata object used to
 //! initialize the class. The number of refinement levels actually 
 //! saved to the data collection are determined by \p reflevels. If
 //! \p reflevels is zero, the default, only the coarsest approximation is
 //! saved. If \p reflevels is one, all the coarsest and first refinement 
 //! level is saved, and so on. A value of -1 indicates the maximum
 //! refinment level permitted by the VDF
 //!
 //! \param[in] timestep Time step of the variable to read
 //! \param[in] varname Name of the variable to read
 //! \param[in] reflevel Refinement level of the variable. A value of -1
 //! indicates the maximum refinment level.
 //! \retval status Returns a non-negative value on success
 //! \sa Metadata::GetVariableNames(), Metadata::GetNumTransforms()
 //!
 virtual int	OpenVariableWrite(
	size_t timestep,
	const char *varname,
	int reflevel = -1
 );

 virtual int OpenVariableRead(
	size_t timestep,
	const char *varname,
	int reflevel = 0,
	int lod = 0
 ) {SetErrMsg("Operation not supported"); return(-1);};


 virtual int	CloseVariable();

 //! Transform and write out two "slabs" of voxels to the currently opened
 //! multiresolution
 //! data volume.  Subsequent calls will write successive pairs of slabs
 //! until the entire volume has been written. The number of transforms
 //! applied is determined by the contents of the Metadata structure
 //! used to initialize this class. If zero, the slabs are not transformed
 //! and are written at their full resolution.
 //! The dimensions of a pair of slabs is NBX by NBY by 2,
 //! where NBX is the dimesion of the volume along the X axis, specified
 //! in blocks, and NBY is the Y axis dimension. The dimension of each block
 //! are given by the Metadata structure used to initialize this class.
 //! It is the caller's responsbility to pad the slabs to block boundaries.
 //!
 //! This method should be called exactly NBZ/2 times, where NBZ is the
 //! dimesion of the volume in blocks along the Z axis. Each invocation
 //! should pass a succesive pair of volume slabs.
 //! 
 //! \sa two_slabs A pair of slabs of raw data
 //! \retval status Returns a non-negative value on success
 //
// int	WriteSlabs(const float *two_slabs);
 int	WriteSlabs(float *two_slabs);

protected:

 void _GetDataRange(float range[2]) const;

private:

 int	slab_cntr_c;
 int	is_open_c;
 float	*lambda_blks_c[MAX_LEVELS];	// temp storage for lambda blocks
 float	*zero_block_c;	// a block of zero data for padding
 size_t _block_size;

 float _dataRange[2];

 int	write_slabs(
	const float *two_slabs,
	int reflevel
	);

 int	write_gamma_slabs(
	int	level,
	const float *two_slabs,
	int	src_nbx,
	int	src_nby,
	float	*dst_lambda_buf,
	int	dst_nbx,
	int	dst_nby
	);

 int	my_realloc();
 void	my_free();

 void	_WaveletBlock3DWriter();

};

};

#endif	//	_WavletBlock3d_h_
