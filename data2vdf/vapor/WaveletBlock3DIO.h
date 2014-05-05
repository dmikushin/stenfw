//
//      $Id: WaveletBlock3DIO.h,v 1.14 2008/06/16 22:24:33 clynejp Exp $
//


#ifndef	_WavletBlock3DIO_h_
#define	_WavletBlock3DIO_h_

#include <cstdio>
#include <netcdf.h>
#include <vapor/MyBase.h>
#include <vapor/WaveletBlock3D.h>
#include <vapor/WaveletBlockIOBase.h>

namespace VAPoR {


//
//! \class WaveletBlock3DIO
//! \brief Performs data IO to VDF files.
//! \author John Clyne
//! \version $Revision: 1.14 $
//! \date    $Date: 2008/06/16 22:24:33 $
//!
//! This class provides an API for performing low-level IO 
//! to/from VDF files
//
class VDF_API	WaveletBlock3DIO : public VAPoR::WaveletBlockIOBase {

public:

 //! Constructor for the WaveletBlock3DIO class.
 //! \param[in] metadata Pointer to a metadata class object for which all
 //! future class operations will apply
 //! \param[in] nthreads Number of execution threads that may be used by
 //! the class for parallel execution.
 //! \note The success or failure of this constructor can be checked
 //! with the GetErrCode() method.
 //!
 //! \sa Metadata, WaveletBlock3DRegionReader, GetErrCode(),
 //
 WaveletBlock3DIO(
	const Metadata *metadata,
	unsigned int	nthreads = 1
 );

 //! Constructor for the WaveletBlock3DIO class.
 //! \param[in] metafile Path to a metadata file for which all
 //! future class operations will apply
 //! \param[in] nthreads Number of execution threads that may be used by
 //! the class for parallel execution.
 //! \note The success or failure of this constructor can be checked
 //! with the GetErrCode() method.
 //!
 //! \sa Metadata, WaveletBlock3DRegionReader, GetErrCode(),
 //
 WaveletBlock3DIO(
	const char *metafile,
	unsigned int	nthreads = 1
 );

 virtual ~WaveletBlock3DIO();


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

 //! Open the named variable for reading
 //!
 //! This method prepares the multiresolution data volume, indicated by a
 //! variable name and time step pair, for subsequent read operations by
 //! methods of this class.  Furthermore, the number of the refinement level
 //! parameter, \p reflevel indicates the resolution of the volume in
 //! the multiresolution hierarchy. The valid range of values for
 //! \p reflevel is [0..max_refinement], where \p max_refinement is the
 //! maximum finement level of the data set: Metadata::GetNumTransforms() - 1.
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
	int reflevel = 0
 );

 //! Close the currently opened variable.
 //!
 //! \sa OpenVariableWrite(), OpenVariableRead()
 //
 virtual int	CloseVariable();

 //! Return the minimum data values for each block in the volume
 //!
 //! This method returns an a pointer to an internal array containing
 //! the minimum data value for each block at the specified refinement
 //! level. The serial array is dimensioned nbx by nby by nbz, where
 //! nbx, nby, and nbz are the dimensions of the volume in blocks
 //! at the requested refinement level.
 //! \param[in] reflevel Refinement level requested. The coarsest
 //! refinement level is 0 (zero). A value of -1 indicates the finest
 //! refinement level contained in the VDC.
 //! \param[out] mins The address of a pointer to float to which will
 //! be assigned the address of the internal min data array
 //!
 //! \nb The values returned are undefined if the variable is either
 //! not open for reading, closed after writing
 //!
 int	GetBlockMins(const float **mins, int reflevel);

 //! Return the maximum data values for each block in the volume
 //!
 int	GetBlockMaxs(const float **maxs, int reflevel);

 //! Unpack a block into a contiguous volume
 //!
 //! Unblock the block \p blk into a volume pointed to by \p voxels
 //! \param[in] blk A block of voxels
 //! \param[in] bcoord Offset of the start of the block within the 
 //! volume in integer coordinates
 //! \param[in] min Minimum extents of destination volume in voxel 
 //! coordinates. Must be between 0 and block_size-1
 //! \param[in] max Maximum extents of destination volume in voxel 
 //! coordinates. 
 //! \param[out] voxels A pointer to a volume
 //
 void	Block2NonBlock(
		const float *blk, 
		const size_t bcoord[3],
		const size_t min[3],
		const size_t max[3],
		float	*voxels
	) const;


protected:

 float	*super_block_c;		// temp storage for gamma blocks;
 VAPoR::WaveletBlock3D	*wb3d_c;

 virtual int	ncDefineDimsVars(
	int j,
	const string &path,
	const int bs_dim_ids[3], 
	const int dim_ids[3]

 );

 virtual int	ncVerifyDimsVars(
	int j,
	const string &path
 );


 // This method moves the file pointer associated with the currently
 // open variable to the disk block containing lambda coefficients
 // indicated by the block coordinates 'bcoord'.
 //
 // A non-negative return value indicates success
 //
 int	seekLambdaBlocks(const size_t bcoord[3]);

 // This method moves the file pointer associated with the currently
 // open variable to the disk block containing gamma coefficients
 // indicated by the block coordinates 'bcoord', and the refinement level
 // 'reflevel'. The parameter 'reflevel'
 // must be in the range [1.._max_reflevel]. Note, if max_xforms_c is 
 // zero, an error is generated as there are no gamma coefficients.
 //
 // A non-negative return value indicates success
 //
 int	seekGammaBlocks(const size_t bcoord[3], int reflevel);

 // Read 'n' contiguous coefficient blocks, associated with the refinement
 // level, 'reflevel', from the currently open variable
 // file. The 'reflevel' parameter must be in the range [0.._max_reflevel],
 // where a value of zero indicates the coefficients for the
 // finest resolution. The results are stored in 'blks', which must 
 // point to an area of adequately sized memory.
 //
 int	readBlocks(size_t n, float *blks, int reflevel);

 //
 // Read 'n' contiguous lambda coefficient blocks
 // from the currently open variable file. 
 // The results are stored in 'blks', which must 
 // point to an area of adequately sized memory.
 //
 int	readLambdaBlocks(size_t n, float *blks);

 //
 // Read 'n' contiguous gamma coefficient blocks, associated with
 // the indicated refinement level, 'ref_level',
 // from the currently open variable file. 
 // The results are stored in 'blks', which must 
 // point to an area of adequately sized memory.
 // An error is generated if 'reflevel' is less than one or
 // 'reflevel' is greater than _max_reflevel.
 //
 int	readGammaBlocks(size_t n, float *blks, int reflevel);

 // Write 'n' contiguous coefficient blocks, associated with the indicated
 // number of transforms, 'num_xforms', from the currently open variable
 // file. The 'num_xforms' parameter must be in the range [0.._max_reflevel],
 // where a value of zero indicates the coefficients for the
 // finest resolution. The coefficients are copied from the memory area
 // pointed to by 'blks'
 //
 int	writeBlocks(const float *blks, size_t n, int reflevel);

 // Write 'n' contiguous lambda coefficient blocks
 // from the currently open variable file. 
 // The blocks are copied to disk from the memory area pointed to 
 // by  'blks'.
 //
 int	writeLambdaBlocks(const float *blks, size_t n);

 // Write 'n' contiguous gamma coefficient blocks, associated with
 // the indicated refinement level, 'ref_level',
 // from the currently open variable file. 
 // The data are copied from the area of memory pointed to by 'blks'.
 // An error is generated if ref_level is less than one or
 // 'ref_level' is greater than _max_reflevel.
 //
 int	writeGammaBlocks(const float *blks, size_t n, int reflevel);



private:
 int	_objInitialized;	// has the obj successfully been initialized?


 int	my_alloc();
 void	my_free();


 int	_WaveletBlock3DIO();

};

}

#endif	//	_WavletBlock3d_h_
