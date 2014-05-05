//
//      $Id: Metadata.h,v 1.38 2010/09/24 17:14:07 southwic Exp $
//


#ifndef	_Metadata_h_
#define	_Metadata_h_

#include <stack>
#include <expat.h>
#include <vapor/MyBase.h>
#include <vapor/common.h>
#include <vapor/XmlNode.h>
#include <vapor/ExpatParseMgr.h>
#ifdef WIN32
#pragma warning(disable : 4251)
#endif

namespace VAPoR {

//
//! \class Metadata
//! \brief An abstract class for managing metadata for a collection 
//! of gridded data. The data collection may support two forms 
//! of data reduction: multi-resolution (a hierarchy of grids, each dimension
//! a factor of two coarser than the preceeding), and level-of-detail (a
//! sequence of one more compressions). 
//!
//! Implementers must derive a concrete class from Metadata to support
//! a particular data collection type
//!
//! \author John Clyne
//! \version $Revision: 1.38 $
//! \date    $Date: 2010/09/24 17:14:07 $
//!
//!
class VDF_API Metadata {
public:

 //! An enum of variable types. Variables defined in a data collection
 //! may be either three-dimensional
 //! (\p VAR3D), or two-dimensional. In the latter case the two-dimesional
 //! data are constrained to lie in the XY, XZ, or YZ coordinate planes
 //! of a 3D volume
 //! 
 enum VarType_T {
	VARUNKNOWN = -1,
	VAR3D, VAR2D_XY, VAR2D_XZ, VAR2D_YZ
 };

 virtual ~Metadata() {};

 //! Get the native dimension of a volume
 //!
 //! Returns the X,Y,Z coordinate dimensions of all data variables
 //! in grid (voxel) coordinates full resolution.
 //!
 //! \param[in] reflevel Refinement level of the variable
 //! \param[out] dim A three element vector (ordered X, Y, Z) containing the 
 //! voxel dimensions of the data at full resolution.
 //!
 //
 virtual void   GetGridDim(size_t dim[3]) const = 0;

 //! Return the internal blocking factor used for data. 
 //!
 //! Returns the X,Y,Z coordinate dimensions of all internal data blocks 
 //! in grid (voxel) coordinates.  If the data are
 //! not blocked this method should return the same values as 
 //! GetDim().
 //!
 //! \retval bs  A three element vector containing the voxel dimension of
 //! a data block
 //
 virtual const size_t *GetBlockSize() const = 0;

 //! Return the internal blocking factor at a given refinement level
 //!
 //! For multi-resolution data this method returns the dimensions
 //! of a data block at refinement level \p reflevel, where reflevel
 //! is in the range 0 to GetNumTransforms(). A value of -1 may be 
 //! specified to indicate the maximum refinement level. In fact,
 //! any value outside the valid refinement level range will be treated
 //! as the maximum refinement level.
 //! 
 //! \param[in] reflevel Refinement level 
 //! \param[bs] dim Transformed dimension.
 //!
 //! \retval bs  A three element vector containing the voxel dimension of
 //! a data block
 //
 virtual void GetBlockSize(size_t bs[3], int reflevel) const = 0;


 //! Return number of transformations in hierarchy 
 //!
 //! For multi-resolution data this method returns the number of
 //! coarsened approximations present. If no approximations 
 //! are available - if only the native data are present - the return
 //! value is 0.
 //!
 //! \retval n  The number of coarsened data approximations available
 //
 virtual int GetNumTransforms() const {return(0); };

 //! Return the compression ratios available.
 //!
 //! For data sets offering level-of-detail, the method returns a 
 //! vector of integers, each specifying an available compression factor.
 //! For example, a factor of 10 indicates a compression ratio of 10:1.
 //! The vector returned is sorted from highest compression ratio
 //! to lowest. I.e. the most compressed data maps to index 0 in
 //! the returned vector.
 //!
 //! \retval cr A vector of one or more compression factors
 //
 virtual vector <size_t> GetCRatios() const {
	vector <size_t> cr; cr.push_back(1); return(cr);
 }


 //! Return the domain extents specified in user coordinates
 //!
 //! Variables in the data represented by spatial coordinates,  such as 
 //! velocity components, are expected to be expressed in the same units 
 //! the returned domain extents.
 //! For data sets with moving (time varying) domains, this method should
 //! return the global bounds of the entire time series.
 //!
 //! \retval extents A six-element array containing the min and max
 //! bounds of the data domain in user-defined coordinates. The first
 //! three elements specify the minimum X, Y, and Z bounds, respectively,
 //! the second three elements specify the maximum bounds.
 //!
 //! \sa GetTSExtents();
 //
 virtual vector<double> GetExtents() const = 0;

 //! Return the number of time steps in the data collection
 //!
 //! \retval value The number of time steps 
 //!
 //
 virtual long GetNumTimeSteps() const = 0;


 //! Return the names of the variables in the collection 
 //!
 //! This method returns a vector of all variables of all types
 //! in the data collection
 //!
 //! \retval value is a space-separated list of variable names
 //!
 //
 virtual vector <string> GetVariableNames() const;

 //! Return the Proj4 map projection string.
 //!
 //! \retval value An empty string if a Proj4 map projection is
 //! not available, otherwise a properly formatted Proj4 projection
 //! string is returned.
 //!
 //
 virtual string GetMapProjection() const {string empty; return (empty); };

 //! Return the names of the 3D variables in the collection 
 //!
 //! \retval value is a space-separated list of 3D variable names.
 //! An emptry string is returned if no variables of this type are present
 //!
 //
 virtual vector <string> GetVariables3D() const = 0;

 //! Return the names of the 2D, XY variables in the collection 
 //!
 //! \retval value is a space-separated list of 2D XY variable names
 //! An emptry string is returned if no variables of this type are present
 //!
 //
 virtual vector <string> GetVariables2DXY() const = 0;

 //! Return the names of the 2D, XZ variables in the collection 
 //!
 //! \retval value is a space-separated list of 2D ZY variable names
 //! An emptry string is returned if no variables of this type are present
 //!
 //
 virtual vector <string> GetVariables2DXZ() const = 0;

 //! Return the names of the 2D, YZ variables in the collection 
 //!
 //! \retval value is a space-separated list of 2D YZ variable names
 //! An emptry string is returned if no variables of this type are present
 //!
 //
 virtual vector <string> GetVariables2DYZ() const = 0;


 //! Return a three-element boolean array indicating if the X,Y,Z
 //! axes have periodic boundaries, respectively.
 //!
 //! \retval boolean-vector  
 //!
 //
 virtual vector<long> GetPeriodicBoundary() const = 0;

 //! Return a three-element integer array indicating the coordinate
 //! ordering permutation.
 //!
 //! \retval integer-vector  
 //!
 virtual vector<long> GetGridPermutation() const = 0;

 //! Return the time for a time step
 //!
 //! This method returns the time, in user-defined coordinates,
 //! associated with the time step, \p ts. Variables such as 
 //! velocity field components that are expressed in distance per 
 //! units of time are expected to use the same time coordinates
 //! as the values returned by this mehtod.
 //!
 //! \param[in] ts A valid data set time step in the range from zero to
 //! GetNumTimeSteps() - 1.
 //!
 //! \retval value The user time at time step \p ts. If \p ts is outside
 //! the valid range zero is returned.
 //!
 //
 virtual double GetTSUserTime(size_t ts) const = 0;


 //! Return the time for a time step
 //!
 //! This method returns the user time, 
 //! associated with the time step, \p ts, as a formatted string. 
 //! The returned time stamp is intended to be used for annotation
 //! purposes
 //!
 //! \param[in] ts A valid data set time step in the range from zero to
 //! GetNumTimeSteps() - 1.
 //! \param[out] s A formated time string. If \p ts is outside
 //! the valid range zero the empty string is returned.
 //!
 //
 virtual void GetTSUserTimeStamp(size_t ts, string &s) const = 0;

 //! Return the domain extents specified in user coordinates
 //! for the indicated time step
 //!
 //! For data collections defined on moving (time varying) domains,
 //! this method returns the domain extents in user coordinates 
 //! for the indicated time step, \p ts.
 //!
 //! \param[in] ts A valid data set time step in the range from zero to
 //! GetNumTimeSteps() - 1.
 //!
 //! \retval extents A six-element array containing the min and max
 //! bounds of the data domain in user-defined coordinates. If \p ts
 //! is outside the valid range the value of GetExtents() is returned.
 //!
 //
 virtual vector<double> GetTSExtents(size_t ) const {
	return(GetExtents());
 }

 //! Get the dimension of a volume
 //!
 //! Returns the X,Y,Z coordinate dimensions of all data variables
 //! in grid (voxel) coordinates at the resolution
 //! level indicated by \p reflevel. Hence, all variables of a given 
 //! type (3D or 2D)
 //! must have the same dimension. If \p reflevel is -1 (or the value
 //! returned by GetNumTransforms()) the native grid resolution is 
 //! returned. In fact, any value outside the valid range is treated
 //! as the maximum refinement level
 //!
 //! \param[in] reflevel Refinement level of the variable
 //! \param[out] dim A three element vector (ordered X, Y, Z) containing the 
 //! voxel dimensions of the data at the specified resolution.
 //!
 //! \sa GetNumTransforms()
 //
 virtual void   GetDim(size_t dim[3], int reflevel = 0) const;

 //! Get dimension of a volume in blocks
 //!
 //! Performs same operation as GetDim() except returns
 //! dimensions in block coordinates instead of voxels.
 //! \param[in] reflevel Refinement level of the variable. A value of -1
 //! indicates the maximum refinment level defined. In fact, any value 
 //! outside the valid range is treated as the maximum refinement level
 //! \param[out] bdim Transformed dimension in blocks.
 //!
 //! \sa Metadata::GetNumTransforms()
 //
 virtual void   GetDimBlk(size_t bdim[3], int reflevel = 0) const; 


 //! Map integer voxel coordinates into integer block coordinates. 
 //! 
 //! Compute the integer coordinate of the block containing
 //! a specified voxel. 
 //! \param[in] vcoord Coordinate of input voxel in integer (voxel)
 //! coordinates
 //! \param[out] bcoord Coordinate of block in integer coordinates containing
 //! the voxel. 
 //! \param[in] reflevel Refinement level of the variable. A value of -1
 //! indicates the maximum refinment level defined.
 //!
 virtual void	MapVoxToBlk(const size_t vcoord[3], size_t bcoord[3], int reflevel = -1) const;
		

 //! Map integer voxel coordinates to user-defined floating point coords.
 //!
 //! Map the integer coordinates of the specified voxel to floating
 //! point coordinates in a user defined space. The voxel coordinates,
 //! \p vcoord0 are specified relative to the refinement level
 //! indicated by \p reflevel for time step \p timestep.  
 //! The mapping is performed by using linear interpolation 
 //! The user-defined coordinate system is obtained
 //! from the Metadata structure passed to the class constructor.
 //! The user coordinates are returned in \p vcoord1.
 //! Results are undefined if vcoord is outside of the volume 
 //! boundary.
 //!
 //! \param[in] timestep Time step of the variable. If an invalid
 //! timestep is supplied the global domain extents are used. 
 //! \param[in] vcoord0 Coordinate of input voxel in integer (voxel)
 //! coordinates
 //! \param[out] vcoord1 Coordinate of transformed voxel in user-defined,
 //! floating point  coordinates
 //! \param[in] reflevel Refinement level of the variable. A value of -1
 //! indicates the maximum refinment level defined for the VDC. In fact,
 //! any invalid value is treated as the maximum refinement level
 //!
 //! \sa Metatdata::GetGridType(), Metadata::GetExtents(), 
 //! GetTSXCoords()
 //
 virtual void	MapVoxToUser(
	size_t timestep,
	const size_t vcoord0[3], double vcoord1[3], int ref_level = 0
 ) const;

 //! Map floating point coordinates to integer voxel offsets.
 //!
 //! Map floating point coordinates, specified relative to a 
 //! user-defined coordinate system, to the closest integer voxel 
 //! coordinates for a voxel at a given refinement level. 
 //! The integer voxel coordinates, \p vcoord1, 
 //! returned are specified relative to the refinement level
 //! indicated by \p reflevel for time step, \p timestep.
 //! The mapping is performed by using linear interpolation 
 //! The user defined coordinate system is obtained
 //! from the Metadata structure passed to the class constructor.
 //! Results are undefined if \p vcoord0 is outside of the volume 
 //! boundary.
 //!
 //! If a user coordinate system is not defined for the specified
 //! time step, \p timestep, the global extents for the VDC will 
 //! be used.
 //!
 //! \param[in] timestep Time step of the variable  If an invalid
 //! timestep is supplied the global domain extents are used.
 //! \param[in] vcoord0 Coordinate of input point in floating point
 //! coordinates
 //! \param[out] vcoord1 Integer coordinates of closest voxel, at the 
 //! indicated refinement level, to the specified point.
 //! integer coordinates
 //! \param[in] reflevel Refinement level of the variable. A value of -1
 //! indicates the maximum refinment level defined for the VDC. In fact,
 //! any invalid value is treated as the maximum refinement level
 //!
 //! \sa Metatdata::GetGridType(), Metadata::GetExtents(), 
 //! GetTSXCoords()
 //
 virtual void	MapUserToVox(
	size_t timestep,
	const double vcoord0[3], size_t vcoord1[3], int reflevel = 0
 ) const;

 //! Map floating point coordinates to integer block offsets.
 //!
 //! Map floating point coordinates, specified relative to a 
 //! user-defined coordinate system, to integer coordinates of the block
 //! containing the point at a given refinement level. 
 //! The integer voxel coordinates, \p vcoord1
 //! are specified relative to the refinement level
 //! indicated by \p reflevel for time step, \p timestep.
 //! The mapping is performed by using linear interpolation 
 //! The user defined coordinate system is obtained
 //! from the Metadata structure passed to the class constructor.
 //! The user coordinates are returned in \p vcoord1.
 //! Results are undefined if \p vcoord0 is outside of the volume 
 //! boundary.
 //!
 //! If a user coordinate system is not defined for the specified
 //! time step, \p timestep, the global extents for the VDC will 
 //! be used.
 //!
 //! \param[in] timestep Time step of the variable.  If an invalid
 //! timestep is supplied the global domain extents are used.
 //! \param[in] vcoord0 Coordinate of input point in floating point
 //! coordinates
 //! \param[out] vcoord1 Integer coordinates of block containing the point
 //! \param[in] reflevel Refinement level of the variable. A value of -1
 //! indicates the maximum refinment level defined for the VDC. In fact,
 //! any invalid value is treated as the maximum refinement level
 //!
 //! \sa Metatdata::GetGridType(), Metadata::GetExtents(), 
 //! GetTSXCoords()
 //
 virtual void	MapUserToBlk(
	size_t timestep,
	const double vcoord0[3], size_t bcoord0[3], int reflevel = 0
 ) const {
	size_t v[3];
	MapUserToVox(timestep, vcoord0, v, reflevel);
	Metadata::MapVoxToBlk(v, bcoord0, reflevel);
 }


 //! Return the variable type for the indicated variable
 //!
 //! This method returns the variable type for the variable 
 //! named by \p varname.
 //!
 //! \param[in] varname A 3D or 2D variable name
 //! \retval type The variable type. The constant VAR
 //
 virtual VarType_T GetVarType(const string &varname) const; 

 //! Return true if indicated region coordinates are valid
 //!
 //! Returns true if the region defined by \p reflevel,
 //! \p min, \p max is valid. I.e. returns true if the indicated
 //! volume subregion is contained within the volume. Coordinates are
 //! specified relative to the refinement level.
 //! 
 //! \param[in] min Minimum region extents in voxel coordinates
 //! \param[in] max Maximum region extents in voxel coordinates
 //! \retval boolean True if region is valid
 //! \param[in] reflevel Refinement level of the variable. A value of -1
 //! indicates the maximum refinment level defined. In fact,
 //! any invalid value is treated as the maximum refinement level. 
 //
 virtual int IsValidRegion(
	const size_t min[3], const size_t max[3], int reflevel = 0
 ) const;

 //! Return true if indicated region coordinates are valid
 //!
 //! Returns true if the region defined by \p reflevel, and the block
 //! coordinates
 //! \p min, \p max are valid. I.e. returns true if the indicated
 //! volume subregion is contained within the volume.
 //! Coordinates are
 //! specified relative to the refinement level.
 //!
 //! \param[in] min Minimum region extents in block coordinates
 //! \param[in] max Maximum region extents in block coordinates
 //! \param[in] reflevel Refinement level of the variable. A value of -1
 //! indicates the maximum refinment level defined. In fact,
 //! any invalid value is treated as the maximum refinement level. 
 //! \retval boolean True if region is valid
 //
 virtual int IsValidRegionBlk(
	const size_t min[3], const size_t max[3], int reflevel = 0
 ) const;


};
};

#endif	//	_Metadata_h_
