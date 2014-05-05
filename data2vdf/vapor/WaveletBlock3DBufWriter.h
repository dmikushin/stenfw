//
//      $Id: WaveletBlock3DBufWriter.h,v 1.10 2010/09/24 17:14:07 southwic Exp $
//


#ifndef	_WavletBlock3DBufWriter_h_
#define	_WavletBlock3DBufWriter_h_

#include <vapor/WaveletBlock3DWriter.h>

namespace VAPoR {

//
//! \class WaveletBlock3DBufWriter
//! \brief A slice-based reader for VDF files
//! \author John Clyne
//! \version $Revision: 1.10 $
//! \date    $Date: 2010/09/24 17:14:07 $
//!
//! This class provides an API for reading volumes
//! from a VDF file one slice at a time.
//
class	VDF_API WaveletBlock3DBufWriter : public WaveletBlock3DWriter {

public:

 //! Constructor for the WaveletBlock3DBufWriter class.
 //! \param[in,out] metadata A pointer to a Metadata structure identifying the
 //! data set upon which all future operations will apply.
 //! \note The success or failure of this constructor can be checked
 //! with the GetErrCode() method.
 //!
 //! \sa MetadataVDC, GetErrCode()
 //
 WaveletBlock3DBufWriter(
	const MetadataVDC &metadata
 );

 //! Constructor for the WaveletBlock3DBufWriter class.
 //! \param[in] metafile Path to a metadata file for which all
 //! future class operations will apply
 //! \note The success or failure of this constructor can be checked
 //! with the GetErrCode() method.
 //!
 //! \sa MetadataVDC, GetErrCode()
 //
 WaveletBlock3DBufWriter(
	const string &metafile
 );

 virtual ~WaveletBlock3DBufWriter();


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

 virtual int	CloseVariable();

 //! Transform and write a single slice of voxels to the current file
 //!
 //! Transform and write a single slice (2D array) of voxels to the variable
 //! indicated by the most recent call to OpenVariableWrite(). 
 //! The number of transforms
 //! applied is determined by the contents of the Metadata structure
 //! used to initialize this class. If zero, the slices are not transformed
 //! and are written at their full resolution.
 //! The dimensions of a slices is NX by NY,
 //! where NX is the dimesion of the volume along the X axis, specified
 //! in voxels, and NY is the Y axis dimension. 
 //!
 //! This method should be called exactly NZ times for each opened variable,
 //! where NZ is the dimension of the volume in voxels along the Z axis. Each
 //! invocation should pass a successive slice of volume data.
 //!
 //! \param[in] slice A slices of volume data
 //! \retval status Returns a non-negative value on success
 //! \sa OpenVariableRead()
 //!
 int	WriteSlice(const float *slice);

protected:

private:

 int	slice_cntr_c;

 float	*buf_c;
 float	*bufptr_c;
 
 int	is_open_c;

 void	_WaveletBlock3DBufWriter();

};

}

#endif	//	WaveletBlock3DBufWriter
