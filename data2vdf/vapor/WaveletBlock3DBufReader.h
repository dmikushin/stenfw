//
//      $Id: WaveletBlock3DBufReader.h,v 1.11 2010/08/18 22:13:24 clynejp Exp $
//


#ifndef	_WavletBlock3DBufReader_h_
#define	_WavletBlock3DBufReader_h_

#include <vapor/WaveletBlock3DReader.h>

namespace VAPoR {

//
//! \class WaveletBlock3DBufReader
//! \brief A slice-based reader for VDF files
//! \author John Clyne
//! \version $Revision: 1.11 $
//! \date    $Date: 2010/08/18 22:13:24 $
//!
//! This class provides an API for reading volumes 
//! from a VDF file one slice at a time.
//
class VDF_API WaveletBlock3DBufReader : public WaveletBlock3DReader {

public:

 //! Constructor for the WaveletBlock3DBufReader class.
 //! \param[in] metadata A pointer to a Metadata structure identifying the
 //! data set upon which all future operations will apply.
 //! \note The success or failure of this constructor can be checked
 //! with the GetErrCode() method.
 //!
 //! \sa Metadata, GetErrCode()
 //
 WaveletBlock3DBufReader(
	const MetadataVDC &metadata
 );

 //! Constructor for the WaveletBlock3DBufReader class.
 //! \param[in] metafile Path to a metadata file for which all
 //! future class operations will apply
 //! \note The success or failure of this constructor can be checked
 //! with the GetErrCode() method.
 //!
 //! \sa Metadata, GetErrCode()
 //
 WaveletBlock3DBufReader(
	const string &metafile
 );

 virtual ~WaveletBlock3DBufReader();


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

 virtual int	CloseVariable();

 //! Read the next volume slice from the currently opened file
 //!
 //! Read in, inverse transform,  and return a slice (2D array) of 
 //! voxels from the currently opened multiresolution data volume.
 //! Subsequent calls will read successive slices
 //! until the entire volume has been read. 
 //! It is the caller's responsibility to ensure that the array pointed 
 //! to by \p slice contains enough space to accomodate
 //! an NX by NY dimensioned slice, where NX is the dimesion of the 
 //! volume along the X axis, specified
 //! in **voxels**, and NY is the Y axis dimension.
 //!
 //! ReadSlice will fail if the requested data are not present. The
 //! VariableExists() method may be used to determine if the data
 //! identified by a (resolution,timestep,variable) tupple are
 //! available on disk.
 //! \param[out] slice The requested volume slice
 //! \retval status Returns a non-negative value on success
 //! \sa OpenVariableRead()
 //!
 //! ReadSlice returns 0 if the entire volume has been read.
 //
 int	ReadSlice(float *slice);


private:

 int	slice_cntr_c;

 float	*buf_c;
 float	*bufptr_c;
 
 int	is_open_c;

 void	_WaveletBlock3DBufReader();

};

}

#endif	//	WaveletBlock3DBufReader
