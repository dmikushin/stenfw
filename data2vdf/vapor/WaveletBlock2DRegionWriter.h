//
//      $Id: WaveletBlock2DRegionWriter.h,v 1.2 2009/02/20 23:02:41 clynejp Exp $
//


#ifndef	_WavletBlock2DRegionWriter_h_
#define	_WavletBlock2DRegionWriter_h_

#include <vapor/MyBase.h>
#include "WaveletBlock2DIO.h"

namespace VAPoR {

//
//! \class WaveletBlock2DRegionWriter
//! \brief A subregion write for VDC files
//! \author John Clyne
//! \version $Revision: 1.2 $
//! \date    $Date: 2009/02/20 23:02:41 $
//!
//! This class provides an API for writing volume sub-regions  
//! to a VDC
//
class VDF_API	WaveletBlock2DRegionWriter : public WaveletBlock2DIO {

public:

 //! Constructor for the WaveletBlock2DRegionWriter class. 
 //!
 //! \param[in] metadata A pointer to a Metadata structure identifying the
 //! data set upon which all future operations will apply. 
 //! \note The success or failure of this constructor can be checked
 //! with the GetErrCode() method.
 //!
 //! \sa Metadata, GetErrCode()
 //
 WaveletBlock2DRegionWriter(
	const Metadata *metadata
 );

 //! Constructor for the WaveletBlock2DRegionWriter class. 
 //!
 //! \param[in] metafile Path to a metadata file for which all
 //! future class operations will apply
 //! \note The success or failure of this constructor can be checked
 //! with the GetErrCode() method.
 //!
 //! \sa Metadata, GetErrCode()
 //
 WaveletBlock2DRegionWriter(
	const char	*metafile
 );

 virtual ~WaveletBlock2DRegionWriter();


 //! Open the named 2D variable for writing
 //!
 //! This method prepares the multiresolution data area, indicated by a
 //! variable name and time step pair, for subsequent write operations by
 //! methods of this class.  
 //! The number of refinement levels actually
 //! saved to the data collection are determined by \p reflevels. If
 //! \p reflevels is zero, the default, only the coarsest approximation is
 //! saved. If \p reflevels is one, the coarsest and first refinement
 //! level is saved, and so on. A value of -1 indicates the maximum
 //! refinment level permitted by the VDF
 //!
 //! An error occurs, indicated by a negative return value, if the
 //! area identified by the {varname, timestep, reflevel} can
 //! not be written . 
 //!
 //! \param[in] timestep Time step of the variable to write
 //! \param[in] varname Name of the variable to write
 //! \param[in] reflevel Refinement level of the variable. A value of -1
 //! indicates the maximum refinment level defined for the VDC
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
	int reflevel = 0
 ) {SetErrMsg("Operation not supported"); return(-1);};


 //! Close the data area opened by the most recent call to 
 //! OpenVariableWrite()
 //! \retval status Returns a non-negative value on success
 //! \sa OpenVariableWrite()
 //!
 virtual int	CloseVariable();

 //! Write a subregion to the currently opened multiresolution
 //! data area.  
 //! The \p min and \p max vectors identify the minium and
 //! maximum extents, in voxel coordinates, of the subregion of interest. The
 //! minimum valid value of 'min' is (0,0), the maximum valid value of
 //! \p max is (nx-1,ny-1), where nx, and ny are the voxel dimensions
 //! of the area at the native resolution (finest resolution).
 //! The area subregion to be written is pointed to
 //! by \p region. 
 //!
 //! \param[in] min Minimum region extents in voxel coordinates
 //! \param[in] max Maximum region extents in voxel coordinates
 //! \param[in] region The volume subregion to write
 //! \retval status Returns a non-negative value on success
 //! \sa OpenVariableWrite(), Metadata::GetDimension()
 //
 int	WriteRegion(
	const float *region,
	const size_t min[2], const size_t max[2]
 );

 //! Write a subregion to the currently opened multiresolution
 //! data area.  
 //!
 //! The data area to be written is pointed to
 //! by \p region. 
 //!
 //! \param[in] region The volume subregion to write
 //! \retval status Returns a non-negative value on success
 //! \sa OpenVariableWrite(), Metadata::GetDimension()
 //
 int	WriteRegion(
	const float *region
 );


 //! Write a area subregion to the currently opened multiresolution
 //! data volume.  
 //!
 //! This method is identical to the WriteRegion() method with the exception
 //! that the region boundaries are defined in tile, not voxel, coordinates.
 //! Secondly, unless the 'tile' parameter  is set, the internal
 //! blocking of the data will be preserved. I.e. the data are assumed
 //! to already be tileed.
 //!
 //! \param[in] bmin Minimum region extents in tile coordinates
 //! \param[in] bmax Maximum region extents in tile coordinates
 //! \param[in] region The volume subregion to write
 //! \param[in] tile If true, tile the data before writing/transforming
 //! \retval status Returns a non-negative value on success
 //! \sa OpenVariableWrite(), Metadata::GetBlockSize(), MapVoxToBlk()
 //
 int	BlockWriteRegion(
	const float *region,
	const size_t bmin[2], const size_t bmax[2], 
	int tile = 1
 );

private:
 int	_objInitialized;	// has the obj successfully been initialized?

 float	*_lambda_tiles[MAX_LEVELS];
 float	*_padblock;

 const float *_regionData;	// Pointer to data passed to WriteRegion() 
 size_t _regMin[2];
 size_t _regMax[2];		// coordinates (in voxels) of region relative to
						// a super-block aligned enclosing region
 size_t _regBSize[2];	// Dimensions (in blocks) of superblock-aligned
						// enclosing region

 size_t _volBMin[2];
 size_t _volBMax[2];	// Bounds (in blocks) of subregion relative to
						// the global volume
 int _is_open;	// open for writing

 // Process a region that requies no transforms
 //
 int _WriteUntransformedRegion(
	const size_t bs[2],
	const size_t min[2],
	const size_t max[2],
	const float *region,
	int tile
 );


 // Copy a tile-sized subvolume to a brick (tile)
 //
 void brickit(
	const float *srcptr,	// ptr to start of volume containing subvolume 
	const size_t bs[2],
	size_t nx, size_t ny, // dimensions of volume
	size_t x, size_t y, 		// voxel coordinates of subvolume
    float *brickptr						// brick destination
 ); 

 // Copy a partial tile-sized subvolume to a brick (tile)
 //
 void brickit(
	const float *srcptr,	// ptr to start of volume containing subvolume 
	const size_t bs[2],
	size_t nx, size_t ny, // dimensions of volume
	size_t srcx,
	size_t srcy,	// voxel coordinates of subvolume
	size_t dstx,
	size_t dsty,	// coordinates within brick (in voxles) for copy
    float *brickptr		// brick destination
 ); 


 void copy_top_superblock(
	const size_t bs[2],
    int srcx,
    int srcy, // coordinates (in tiles) of superblock within 
				// superblock-aligned enclosing region.
	float *dst_super_block	// destination super block
 );

 // Recursively coarsen a data quadrant
 //
 int process_quadrant(
	size_t sz,			// dimension of quadrant in tiles
	int srcx,			
	int srcy,			// Coordinates of quadrant (in tiles) relative to vol
	int dstx,
	int dsty,			// Coordinates (in tiles) of subregion destination
	int quad			// quadrant indicator (0-3)
 );

 // compute the min and max of a tile
 void compute_minmax(
	const float *tileptr,
	size_t bx, size_t by,
	int level
 );


 int	my_alloc(); 
 void	my_free(); 

 int	_WriteRegion(
	const float *region,
	const size_t min[2], const size_t max[2], 
	int tile
 );

 int	_WaveletBlock2DRegionWriter();

};

};

#endif	//	_WavletBlock2DRegionWriter_h_
