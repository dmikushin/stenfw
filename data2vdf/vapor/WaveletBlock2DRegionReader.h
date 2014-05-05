//
//      $Id: WaveletBlock2DRegionReader.h,v 1.3 2009/03/06 14:24:38 alannorton Exp $
//


#ifndef	_WavletBlock2DRegionReader_h_
#define	_WavletBlock2DRegionReader_h_

#include <vapor/MyBase.h>
#include "WaveletBlock2DIO.h"

namespace VAPoR {

//
//! \class WaveletBlock2DRegionReader
//! \brief A sub-region reader for VDF files
//! \author John Clyne
//! \version $Revision: 1.3 $
//! \date    $Date: 2009/03/06 14:24:38 $
//!
//! This class provides an API for extracting area sub-regions  
//! from a VDF file
//
class VDF_API	WaveletBlock2DRegionReader : public WaveletBlock2DIO {

public:

 //! Constructor for the WaveletBlock2DRegionReader class. 
 //! \param[in] metadata A pointer to a Metadata structure identifying the
 //! data set upon which all future operations will apply. 
 //! \note The success or failure of this constructor can be checked
 //! with the GetErrCode() method.
 //!
 //! \sa Metadata, GetErrCode()
 //
 WaveletBlock2DRegionReader(
	const Metadata *metadata
 );

 //! Constructor for the WaveletBlock2DRegionReader class. 
 //! \param[in] metafile Path to a metadata file for which all
 //! future class operations will apply
 //! \note The success or failure of this constructor can be checked
 //! with the GetErrCode() method.
 //!
 //! \sa Metadata, GetErrCode()
 //
 WaveletBlock2DRegionReader(
	const char	*metafile
 );

 virtual ~WaveletBlock2DRegionReader();


 //! Open the named variable for reading
 //!
 //! This method prepares the multiresolution data area, indicated by a
 //! variable name and time step pair, for subsequent read operations by
 //! methods of this class.  Furthermore, the number of the refinement level
 //! parameter, \p reflevel indicates the resolution of the area in
 //! the multiresolution hierarchy. The valid range of values for
 //! \p reflevel is [0..max_refinement], where \p max_refinement is the
 //! maximum finement level of the data set: Metadata::GetNumTransforms() - 1.
 //! area when the area was created. A value of zero indicates the
 //! coarsest resolution data, a value of \p max_refinement indicates the
 //! finest resolution data.
 //!
 //! An error occurs, indicated by a negative return value, if the
 //! area identified by the {varname, timestep, reflevel} tripple
 //! is not present on disk. Note the presence of a area can be tested
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
	int reflevel = 0
 );
 virtual int OpenVariableWrite(
	size_t /*timestep*/,
	const char * /*varname*/,
	int /* reflevel */ = 0
 ) {SetErrMsg("Operation not supported"); return(-1);};



 //! Close the data area opened by the most recent call to 
 //! OpenVariableRead()
 //! \retval status Returns a non-negative value on success
 //! \sa OpenVariableRead()
 //!
 virtual int	CloseVariable();

 //! Read in and return a subregion from the currently opened multiresolution
 //! data area.  
 //! The \p min and \p max vectors identify the minium and
 //! maximum extents, in voxel coordinates, of the subregion of interest. The
 //! minimum valid value of 'min' is (0,0,0), the maximum valid value of
 //! \p max is (nx-1,ny-1,nz-1), where nx, ny, and nz are the voxel dimensions
 //! of the area at the resolution indicated by \p num_xforms. I.e. 
 //! the coordinates are specified relative to the desired area 
 //! resolution. The area
 //! returned is stored in the memory region pointed to by \p region. It 
 //! is the caller's responsbility to ensure adequate space is available.
 //!
 //! ReadRegion will fail if the requested data are not present. The
 //! VariableExists() method may be used to determine if the data
 //! identified by a (resolution,timestep,variable) tupple are 
 //! available on disk.
 //! \param[in] min Minimum region extents in voxel coordinates
 //! \param[in] max Maximum region extents in voxel coordinates
 //! \param[out] region The requested area subregion
 //! \retval status Returns a non-negative value on success
 //! \sa OpenVariableRead(), Metadata::GetDimension()
 //
 int	ReadRegion(
	const size_t min[2], const size_t max[2], 
	float *region
 );

 //! Read in and return currently opened multiresolution
 //! 2D data area.  
 //!
 //! The area
 //! returned is stored in the memory region pointed to by \p region. It 
 //! is the caller's responsbility to ensure adequate space is available.
 //!
 //! ReadRegion will fail if the requested data are not present. The
 //! VariableExists() method may be used to determine if the data
 //! identified by a (resolution,timestep,variable) tupple are 
 //! available on disk.
 //! \param[out] region The requested area subregion
 //! \retval status Returns a non-negative value on success
 //! \sa OpenVariableRead(), Metadata::GetDimension()
 //
 int	ReadRegion(
	float *region
 );


 //! Read in and return a subregion from the currently opened multiresolution
 //! data area.  
 //!
 //! This method is identical to the ReadRegion() method with the exception
 //! that the region boundaries are defined in tile, not voxel, coordinates.
 //! Secondly, unless the 'untile' parameter  is set, the internal
 //! tiling of the data will be preserved. 
 //!
 //! BlockReadRegion will fail if the requested data are not present. The
 //! VariableExists() method may be used to determine if the data
 //! identified by a (resolution,timestep,variable) tupple are 
 //! available on disk.
 //! \param[in] bmin Minimum region extents in tile coordinates
 //! \param[in] bmax Maximum region extents in tile coordinates
 //! \param[out] region The requested area subregion
 //! \param[in] untile If true, untile the data before copying to \p region
 //! \retval status Returns a non-negative value on success
 //! \sa OpenVariableRead(), Metadata::GetBlockSize(), MapVoxToBlk()
 //
 int	BlockReadRegion(
	const size_t bmin[2], const size_t bmax[2], 
	float *region, int untile = 1
 );

private:
 int	_objInitialized;	// has the obj successfully been initialized?

 float	*_lambda_tiles[MAX_LEVELS];
 float	*_gamma_tiles[MAX_LEVELS];

 int	row_inv_xform(
	const float *lambda_row, 
	unsigned int ljx0, unsigned int ljy0,
	unsigned int ljnx, unsigned int j, float *region, 
	const size_t min[2], const size_t max[2], unsigned int level, int untile
	);
 int	my_realloc(); 
 void	my_free(); 

 int	_ReadRegion(
	const size_t min[2], const size_t max[2], 
	float *region, int untile = 1
 );

 void	_WaveletBlock2DRegionReader();

};

};

#endif	//	_WavletBlock2DRegionReader_h_
