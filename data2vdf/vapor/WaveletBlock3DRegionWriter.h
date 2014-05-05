//
//      $Id: WaveletBlock3DRegionWriter.h,v 1.6 2010/08/18 22:13:24 clynejp Exp $
//


#ifndef	_WavletBlock3DRegionWriter_h_
#define	_WavletBlock3DRegionWriter_h_

#include <vapor/MyBase.h>
#include "WaveletBlockIOBase.h"

namespace VAPoR {

//
//! \class WaveletBlock3DRegionWriter
//! \brief A subregion write for VDC files
//! \author John Clyne
//! \version $Revision: 1.6 $
//! \date    $Date: 2010/08/18 22:13:24 $
//!
//! This class provides an API for writing volume sub-regions  
//! to a VDC
//
class VDF_API	WaveletBlock3DRegionWriter : public WaveletBlockIOBase {

public:

 //! Constructor for the WaveletBlock3DRegionWriter class. 
 //!
 //! \param[in] metadata A pointer to a Metadata structure identifying the
 //! data set upon which all future operations will apply. 
 //! \note The success or failure of this constructor can be checked
 //! with the GetErrCode() method.
 //!
 //! \sa MetadataVDC, GetErrCode()
 //
 WaveletBlock3DRegionWriter(
	const MetadataVDC &metadata
 );

 //! Constructor for the WaveletBlock3DRegionWriter class. 
 //!
 //! \param[in] metafile Path to a metadata file for which all
 //! future class operations will apply
 //! \note The success or failure of this constructor can be checked
 //! with the GetErrCode() method.
 //!
 //! \sa MetadataVDC, GetErrCode()
 //
 WaveletBlock3DRegionWriter(
	const string &metafile
 );

 virtual ~WaveletBlock3DRegionWriter();


 //! Open the named variable for writing
 //!
 //! This method prepares the multiresolution data volume, indicated by a
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
 //! volume identified by the {varname, timestep, reflevel} can
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
	int reflevel = 0,
	int lod = 0
 ) {SetErrMsg("Operation not supported"); return(-1);};



 //! Close the data volume opened by the most recent call to 
 //! OpenVariableWrite()
 //! \retval status Returns a non-negative value on success
 //! \sa OpenVariableWrite()
 //!
 virtual int	CloseVariable();

 //! Write a subregion to the currently opened multiresolution
 //! data volume.  
 //! The \p min and \p max vectors identify the minium and
 //! maximum extents, in voxel coordinates, of the subregion of interest. The
 //! minimum valid value of 'min' is (0,0,0), the maximum valid value of
 //! \p max is (nx-1,ny-1,nz-1), where nx, ny, and nz are the voxel dimensions
 //! of the volume at the native resolution (finest resolution).
 //! The volume subregion to be written is pointed to
 //! by \p region. 
 //!
 //! \param[in] min Minimum region extents in voxel coordinates
 //! \param[in] max Maximum region extents in voxel coordinates
 //! \param[in] region The volume subregion to write
 //! \retval status Returns a non-negative value on success
 //! \sa OpenVariableWrite()
 //
 int	WriteRegion(
	const float *region,
	const size_t min[3], const size_t max[3]
 );

 int	WriteRegion(
	const float *region
 );


 //! Write a volume subregion to the currently opened multiresolution
 //! data volume.  
 //!
 //! This method is identical to the WriteRegion() method with the exception
 //! that the region boundaries are defined in block, not voxel, coordinates.
 //! Secondly, unless the 'block' parameter  is set, the internal
 //! blocking of the data will be preserved. I.e. the data are assumed
 //! to already be blocked.
 //!
 //! \param[in] bmin Minimum region extents in block coordinates
 //! \param[in] bmax Maximum region extents in block coordinates
 //! \param[in] region The volume subregion to write
 //! \param[in] block If true, block the data before writing/transforming
 //! \retval status Returns a non-negative value on success
 //! \sa OpenVariableWrite(), Metadata::GetBlockSize(), MapVoxToBlk()
 //
 int	BlockWriteRegion(
	const float *region,
	const size_t bmin[3], const size_t bmax[3], 
	int block = 1
 );

protected:

 void _GetDataRange(float range[2]) const;
 void _GetValidRegion(size_t minreg[3], size_t maxreg[3]) const; 

private:

 float	*_lambda_blks[MAX_LEVELS];
 float	*_lambda_tiles[MAX_LEVELS];
 float	*_padblock3d;
 float	*_padblock2d;
 size_t _block_size;

 const float *_regionData;	// Pointer to data passed to WriteRegion() 
 size_t _regMin[3];
 size_t _regMax[3];		// coordinates (in voxels) of region relative to
						// a super-block aligned enclosing region

 size_t _validRegMin[3];
 size_t _validRegMax[3];    // Bounds (in voxels) of valid region relative
                            // to the finest level


 size_t _volBMin[3];
 size_t _volBMax[3];	// Bounds (in blocks) of subregion relative to
						// the global volume
 int _is_open;	// open for writing

 float _dataRange[2];       // min and max range of data;

 bool _firstWrite;	// True first time Write() is called for an open var


 // Process a region that requies no transforms
 //
 int _WriteUntransformedRegion3D(
	const size_t min[3],
	const size_t max[3],
	const float *region,
	int block
 );
 int _WriteUntransformedRegion2D(
	const size_t bs[2],
	const size_t min[2],
	const size_t max[2],
	const float *region,
	int block
 );


 // Copy a block-sized subvolume to a brick (block)
 //
 void brickit3d(
	const float *srcptr,	// ptr to start of volume containing subvolume 
	size_t nx, size_t ny, size_t nz, 	// dimensions of volume
	size_t x, size_t y, size_t z,		// voxel coordinates of subvolume
    float *brickptr						// brick destination
 ); 

 // Copy a partial block-sized subvolume to a brick (block)
 //
 void brickit3d(
	const float *srcptr,	// ptr to start of volume containing subvolume 
	size_t nx, size_t ny, size_t nz, 	// dimensions of volume
	size_t srcx,
	size_t srcy,
	size_t srcz,		// voxel coordinates of subvolume
	size_t dstx,
	size_t dsty,
	size_t dstz,		// coordinates within brick (in voxles) for copy
    float *brickptr		// brick destination
 ); 

 // Copy a tile-sized subvolume to a brick (tile)
 //
 void brickit2d(
	const float *srcptr,    // ptr to start of volume containing subvolume
	const size_t bs[2],
	size_t nx, size_t ny, // dimensions of volume
	size_t x, size_t y,         // voxel coordinates of subvolume
	float *brickptr                     // brick destination
 );

 // Copy a partial tile-sized subvolume to a brick (tile)
 //
 void brickit2d(
	const float *srcptr,    // ptr to start of volume containing subvolume
	const size_t bs[2],
	size_t nx, size_t ny, // dimensions of volume
	size_t srcx,
	size_t srcy,    // voxel coordinates of subvolume
	size_t dstx,
	size_t dsty,    // coordinates within brick (in voxles) for copy
	float *brickptr     // brick destination
 );


 void copy_top_superblock3d(
    int srcx,
    int srcy,
    int srcz, // coordinates (in blocks) of superblock within 
				// superblock-aligned enclosing region.
	float *dst_super_block	// destination super block
 );

 void copy_top_superblock2d(
	const size_t bs[2],
	int srcx,
	int srcy, // coordinates (in tiles) of superblock within
				// superblock-aligned enclosing region.
	float *dst_super_block  // destination super block
 );


 // Recursively coarsen a data octant
 //
 int process_octant(
	size_t sz,			// dimension of octant in blocks
	int srcx,			
	int srcy,
	int srcz,			// Coordinates of octant (in blocks) relative to vol
	int dstx,
	int dsty,
	int dstz,			// Coordinates (in blocks) of subregion destination
	int oct			// octant indicator (0-7)
 );


 // Recursively coarsen a data quadrant
 //
 int process_quadrant(
	size_t sz,          // dimension of quadrant in tiles
	int srcx,
	int srcy,           // Coordinates of quadrant (in tiles) relative to vol
	int dstx,
	int dsty,           // Coordinates (in tiles) of subregion destination
	int quad            // quadrant indicator (0-3)
 );

 // compute the min and max of a block
 void compute_minmax3d(
	const float *blkptr,
	size_t bx, size_t by, size_t bz,
	int level
 );

 // compute the min and max of a tile
 void compute_minmax2d(
	const float *tileptr,
	size_t bx, size_t by,
	int level
 );

 int	my_alloc3d(); 
 int	my_alloc2d(); 
 void	my_free(); 

 int _WriteRegion3D(
	const float *region,
	const size_t min[3], const size_t max[3], 
	int block
 );

 int _WriteRegion2D(
	const float *region,
	const size_t min[2],
	const size_t max[2],
	int block
 );

 void _CloseVariable3D();
 void _CloseVariable2D();

 int	_WaveletBlock3DRegionWriter();

};

};

#endif	//	_WavletBlock3DRegionWriter_h_
