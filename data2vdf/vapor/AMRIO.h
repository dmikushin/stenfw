//
//      $Id: AMRIO.h,v 1.7 2010/06/07 16:34:48 clynejp Exp $
//
//***********************************************************************
//                                                                      *
//                      Copyright (C)  2006	                        *
//          University Corporation for Atmospheric Research             *
//                      All Rights Reserved                             *
//                                                                      *
//***********************************************************************
//
//	File:		AMRIO.h
//
//	Author:		John Clyne
//			National Center for Atmospheric Research
//			PO 3000, Boulder, Colorado
//
//	Date:		Thu Jan 5 16:57:43 MST 2006
//
//	Description:	
//
//
#ifndef	_AMRIO_h_
#define	_AMRIO_h_

#include <cstdio>
#include <netcdf.h>
#include <vapor/MyBase.h>
#include <vapor/AMRTree.h>
#include <vapor/AMRData.h>
#include <vapor/VDFIOBase.h>

namespace VAPoR {


//
//! \class AMRIO
//! \brief Performs data IO to VDF files.
//! \author John Clyne
//! \version $$
//! \date    $$
//!
//! This class provides an API for performing IO  on AMR data sets
//! to/from Vapor Data Collections (VDCs)
//
class VDF_API	AMRIO : public VAPoR::VDFIOBase {

public:

 //! Constructor for the AMRIO class.
 //!
 //! \param[in] metadata Pointer to a metadata class object for which all
 //! future class operations will apply. The metadata class object
 //! identifies the VDC for all subsequent data operations
 //! \param[in] nthreads Number of execution threads that may be used by
 //! the class for parallel execution.
 //! \note The success or failure of this constructor can be checked
 //! with the GetErrCode() method.
 //!
 //! \sa MetadataVDC, GetErrCode(),
 //
 AMRIO(
	const MetadataVDC &metadata
 );

 //! Constructor for the AMRIO class.
 //!
 //! \param[in] metafile Path to a metadata file for which all
 //! future class operations will apply. The metadata class object
 //! identifies the VDC for all subsequent data operations
 //! \note The success or failure of this constructor can be checked
 //! with the GetErrCode() method.
 //!
 //! \sa MetadataVDC, GetErrCode(),
 //
 AMRIO(
	const string &metafile
 );

 virtual ~AMRIO();

 //! Returns true if indicated AMR grid exists on disk
 //!
 //! Returns true if the variable identified by the timestep, variable
 //! name, and refinement level is present on disk. Returns 0 if
 //! the variable is not present.
 //! \param[in] ts A valid time step from the Metadata object used
 //! to initialize the class
 //! \param[in] varname A valid variable name
 //! \param[in] reflevel Refinement level requested. The coarsest
 //! refinement level is 0 (zero). A value of -1 indicates the finest
 //! refinement level contained in the VDC.
 //
 int    VariableExists(
	size_t ts,
	const char *varname,
	int reflevel = 0,
	int lod = 0
 ) const ;

 //! Open the specified AMR octree for writing
 //!
 //! Prepare a VDC for the storage of an AMR grid topology octree.
 //! AMR grids contain two components: a topology octree which describes
 //! the cell refinement hierarchy, and the field values associated with
 //! the cells in the octree. A single octree is associated with each
 //! time step, whereas multiple field variables may be defined for
 //! each time step, all associated with the same octree.
 //!
 //! \param[in] timestep Time step of the octree to write
 //! \retval status Returns a non-negative value on success
 //! \sa Metadata::GetVariableNames(), Metadata::GetNumTransforms()
 //! \sa TreeWrite()
 //
 int	OpenTreeWrite(
	size_t timestep
 );

 //! Open the specified AMR octree for reading
 //!
 //! Prepare a VDC for the reading of an AMR grid topology octree.
 //! AMR grids contain two components: a topology octree which describes
 //! the cell refinement hierarchy, and the field values associated with
 //! the cells in the octree. A single octree is associated with each
 //! time step, whereas multiple field variables may be defined for
 //! each time step, all associated with the same octree.
 //!
 //! \param[in] timestep Time step of the octree to write
 //! \retval status Returns a non-negative value on success
 //! \sa Metadata::GetVariableNames(), Metadata::GetNumTransforms()
 //
 int	OpenTreeRead(
	size_t timestep
 );

 //! Close the currently opened octree.
 //!
 //! \sa OpenTreeWrite(), OpenTreeRead()
 //
 int	CloseTree();

 //! Read an AMR octree
 //!
 //! Read the currently opened octree into the AMRTree structure
 //! pointed to \p tree. 
 //!
 //! \param[out] tree Upon success the octree pointed to by \p tree will 
 //! contain the octree associated
 //! with the currently opened time step. 
 //! \sa OpenTreeRead()
 //
 int	TreeRead(AMRTree *tree);


 //! Write an AMR octree
 //!
 //! Write the octree pointed to by \p tree to the VDC at the time step
 //! associated with the currently opened tree.
 //!
 //! \param[in] tree A pointer to an AMR octree
 //! \sa OpenTreeRead()
 int	TreeWrite(const AMRTree *tree);


 //! Open the named AMR variable for writing
 //!
 //! Prepare a VDC for the storage of an AMR grid
 //! via subsequent write operations.
 //! The AMR grid is identified by the specfied time step and
 //! variable name. The maximum number of refinement level
 //! is determined by the Metadata object used to
 //! initialize the class. The number of refinement levels actually 
 //! saved to the data collection are determined by \p reflevels. If
 //! \p reflevels is zero, the default, only the coarsest approximation is
 //! saved. If \p reflevels is one, the coarsest and first refinement 
 //! level is saved, and so on. A value of -1 indicates the maximum
 //! refinment level permitted by the associated Metadata object
 //!
 //! \param[in] timestep Time step of the variable to read
 //! \param[in] varname Name of the variable to read
 //! \param[in] reflevel Refinement level of the variable. A value of -1
 //! indicates the maximum refinment level.
 //! \retval status Returns a non-negative value on success
 //! \sa Metadata::GetVariableNames(), Metadata::GetNumTransforms()
 //!
 int	OpenVariableWrite(
	size_t timestep,
	const char *varname,
	int reflevel = -1
 );
 
 //! Open the named AMR grid for reading
 //!
 //! This method prepares the AMR data grid, indicated by a
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
 int	OpenVariableRead(
	size_t timestep,
	const char *varname,
	int reflevel = 0,
	int lod = 0
 );

 //! Close the currently opened variable.
 //!
 //! \sa OpenVariableWrite(), OpenVariableRead()
 //
 virtual int	CloseVariable();

 //! Read an AMR grid
 //!
 //! Read the currently opened AMR grid into the AMRData structure
 //! pointed to \p data. 
 //!
 //! \param[out] data Upon success the AMR grid pointed to by \p data will 
 //! contain the AMR Grid associated
 //! with the currently opened time step and variable. 
 //! \sa OpenDataRead()
 //
 int	VariableRead(AMRData *data);

 //! Write an AMR grid
 //!
 //! Write the AMR grid pointed to by \p data to the VDC at the time step
 //! associated and variable with the currently opened grid.
 //!
 //! \param[in] tree A pointer to an AMR octree
 //! \sa OpenDataRead()
 int	VariableWrite(AMRData *data);


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
 int	GetBlockMins(const float **mins, int reflevel) const;

 //! Return the maximum data values for each block in the volume
 //!
 int	GetBlockMaxs(const float **maxs, int reflevel) const;

 const float *GetDataRange() const {return (_dataRange);}

 void GetValidRegion(
    size_t min[3], size_t max[3], int reflevel
 ) const;


private:
 typedef int int32_t;

 static const int MAX_LEVELS = 32;	// Max # of refinement leveles permitted

 int	_reflevel;	// refinement level of currently opened file.
						
 string _varName;		// Currently opened variable
 string	_treeFileName;	// Currenly opened tree file name
 string	_dataFileName;	// Currenly opened amr data file name

 float	*_mins[MAX_LEVELS];	// min value contained in a block
 float	*_maxs[MAX_LEVELS];	// max value contained in a block

 int	_treeIsOpen;	// true if an AMR tree file is open
 int	_dataIsOpen;	// true if an AMR data file is open
 int	_treeWriteMode;	// true if file opened for writing
 int	_dataWriteMode;	// true if file opened for writing
 float	_dataRange[2];
 size_t _validRegMin[3];
 size_t _validRegMax[3];


 int _AMRIO();
 int mkpath(size_t timestep, string *path) const;
 int mkpath(size_t timestep, const char *varname, string *path) const;

};

}

#endif	//	_WavletBlock3d_h_
