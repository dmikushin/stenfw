//
//      $Id: WaveletBlock3DReader.h,v 1.11 2010/06/07 16:34:48 clynejp Exp $
//


#ifndef	_WavletBlock3DReader_h_
#define	_WavletBlock3DReader_h_

#include <vapor/MyBase.h>
#include "WaveletBlockIOBase.h"

namespace VAPoR {

//
//! \class WaveletBlock3DReader
//! \brief A slab reader for VDF files
//! \author John Clyne
//! \version $Revision: 1.11 $
//! \date    $Date: 2010/06/07 16:34:48 $
//!
//! This class provides a low-level API for reading data volumes from 
//! a VDF file. The Read methods contained within are the  most efficient
//! (both in terms of memory and performance) for reading an entire data
//! volume.
//
class	VDF_API WaveletBlock3DReader : public WaveletBlockIOBase {

public:

 //! Constructor for the WaveletBlock3DReader class.
 //! \param[in] metadata A pointer to a Metadata structure identifying the
 //! data set upon which all future operations will apply.
 //! \note The success or failure of this constructor can be checked
 //! with the GetErrCode() method.
 //!
 //! \sa Metadata, GetErrCode()
 //
 WaveletBlock3DReader(
	const MetadataVDC &metadata
);

 //! Constructor for the WaveletBlock3DReader class.
 //! \param[in] metadata Path to a metadata file for which all
 //! future class operations will apply
 //! \note The success or failure of this constructor can be checked
 //! with the GetErrCode() method.
 //!
 //! \sa MetadataVDC, GetErrCode()
 //
 WaveletBlock3DReader(
	const string &metafile
 );

 virtual ~WaveletBlock3DReader();

 //! Open the named variable for reading
 //!
 //! This method prepares the multiresolution data volume, indicated by a
 //! variable name and time step pair, for subsequent read operations by
 //! methods of this class.  Furthermore, the number of the refinement level
 //! parameter, \p reflevel indicates the resolution of the volume in
 //! the multiresolution hierarchy. The valid range of values for
 //! \p reflevel is [0..max_refinement], where \p max_refinement is the
 //! maximum finement level of the data set: Metadata::GetNumTransforms().
 //! volume when the volume was created. A value of zero indicates the
 //! coarsest resolution data, a value of \p max_refinement indicates the
 //! finest resolution data.
 //!
 //! An error occurs, indicated by a negative return value, if the
 //! volume identified by the {varname, timestep, reflevel} tripple
 //! is not present on disk. Note the presence of a volume can be tested
 //! for with the VariableExists() method.
 //! \param[in] timestep Time step of the variable to read
 //! \param[in] varname Name of the variable to read
 //! \param[in] reflevel Refinement level of the variable. A value of -1
 //! indicates the maximum refinment level defined for the VDC
 //! \retval status Returns a non-negative value on success
 //! \sa Metadata::GetVariableNames(), Metadata::GetNumTransforms()
 //!
 virtual int	OpenVariableRead(
	size_t timestep,
	const char *varname,
	int reflevel = 0,
	int lod = 0
 );

 virtual int OpenVariableWrite(
	size_t timestep,
	const char *varname,
	int reflevel = 0
 ) {SetErrMsg("Operation not supported"); return(-1);};


 virtual int	CloseVariable();


 //! Read in and return two "slabs" of voxels from the currently opened 
 //! multiresolution
 //! data volume.  Subsequent calls will read successive pairs of slabs
 //! until the entire volume has been read. 
 //! The dimensions of a pair of slabs is NBX by NBY by 2,
 //! where NBX is the dimesion of the volume along the X axis, specified
 //! in **blocks**, and NBY is the Y axis dimension. 
 //! The dimension of each block
 //! are given by the Metadata structure used to initialize this class.
 //! The volume
 //! returned is stored in the memory region pointed to by \p region. It
 //! is the caller's responsbility to ensure adequate space is available.
 //! If the parameter \p unblock is not true, data are returned in
 //! the native storage order used internally by this class. Generally,
 //! it is preferable to set this flag to true.
 //!
 //! ReadRegion will fail if the requested data are not present. The
 //! VariableExists() method may be used to determine if the data
 //! identified by a (resolution,timestep,variable) tupple are
 //! available on disk.
 //!
 //! ReadRegion returns 0 if the entire volume has been read.
 //!
 //! \param[out] two_slabs Next two successive slabs of data
 //! \param[in] unblock If true, unblock the data from it's native storage
 //! format before returning it.
 //! \retval status Returns a non-negative value on success
 //!
 //
 int	ReadSlabs(float *two_slabs, int unblock);

protected:
 void _GetDataRange(float range[2]) const {};

private:

 float	*lambda_blks_c[MAX_LEVELS];	// temp storage for lambda blocks
 float	*scratch_block_c;	// scratch space
 int	slab_cntr_c;
 size_t _block_size;

 int	read_slabs(
	int level,
	const float *src_lambda_buf,
	int src_nbx,
	int src_nby,
	float *two_slabs,
	int dst_nbx,
	int dst_nby
	);

 int	my_realloc();
 void	my_free();

 void	_WaveletBlock3DReader();

};

};

#endif	//	_WavletBlock3DReader_h_
