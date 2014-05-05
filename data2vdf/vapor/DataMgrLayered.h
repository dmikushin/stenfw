//
//      $Id: DataMgrLayered.h,v 1.7 2010/09/24 17:14:07 southwic Exp $
//

#ifndef	_DataMgrLayered_h_
#define	_DataMgrLayered_h_


#include <vector>
#include <string>
#include <vapor/WaveletBlock3DRegionReader.h>
#include <vapor/LayeredIO.h>
#include <vapor/common.h>

namespace VAPoR {

//
//! \class DataMgrLayered
//! \brief A cache based data reader
//! \author John Clyne
//! \version $Revision: 1.7 $
//! \date    $Date: 2010/09/24 17:14:07 $
//!
//! This class provides a wrapper to the Layered()
//! and WaveletBlock2DRegionReader()
//! classes that includes a memory cache. Data regions read from disk through 
//! this
//! interface are stored in a cache in main memory, where they may be
//! be available for future access without reading from disk.
//
class VDF_API DataMgrLayered : public LayeredIO, WaveletBlock3DRegionReader {

public:

 DataMgrLayered(
	const string &metafile,
	size_t mem_size
 );

 DataMgrLayered(
	const MetadataVDC &metadata,
	size_t mem_size
 );


 virtual ~DataMgrLayered() {CloseVariableNative();};

 virtual int _VariableExists(
	size_t ts,
	const char *varname,
	int reflevel = 0,
	int lod = 0
 ) const {
	return (WaveletBlock3DRegionReader::VariableExists(ts,varname,reflevel,lod));
 };

 //
 //	Metadata methods
 //

 virtual const size_t *GetBlockSize() const {
	return(WaveletBlock3DRegionReader::GetBlockSize());
 }

 virtual void GetBlockSize(size_t bs[3], int reflevel) const {
	WaveletBlock3DRegionReader::GetBlockSize(bs, reflevel);
 }

 virtual int GetNumTransforms() const {
	return(WaveletBlock3DRegionReader::GetNumTransforms());
 };

 virtual vector<double> GetExtents() const {
	return(WaveletBlock3DRegionReader::GetExtents());
 };

 virtual long GetNumTimeSteps() const {
	return(WaveletBlock3DRegionReader::GetNumTimeSteps());
 };

 virtual vector <string> _GetVariables3D() const {
	return(WaveletBlock3DRegionReader::GetVariables3D());
 };

 virtual vector <string> _GetVariables2DXY() const {
	return(WaveletBlock3DRegionReader::GetVariables2DXY());
 };
 virtual vector <string> _GetVariables2DXZ() const {
	return(WaveletBlock3DRegionReader::GetVariables2DXZ());
 };
 virtual vector <string> _GetVariables2DYZ() const {
	return(WaveletBlock3DRegionReader::GetVariables2DYZ());
 };

 virtual vector<long> GetPeriodicBoundary() const {
	return(WaveletBlock3DRegionReader::GetPeriodicBoundary());
 };

 virtual vector<long> GetGridPermutation() const {
	return(WaveletBlock3DRegionReader::GetGridPermutation());
 };

 virtual double GetTSUserTime(size_t ts) const {
	return(WaveletBlock3DRegionReader::GetTSUserTime(ts));
 };

 virtual void GetTSUserTimeStamp(size_t ts, string &s) const {
    WaveletBlock3DRegionReader::GetTSUserTimeStamp(ts,s);
 }

 virtual vector<double> GetTSExtents(size_t ts) const {
	return(WaveletBlock3DRegionReader::GetTSExtents(ts));
 };

 virtual string GetMapProjection() const {
	return(WaveletBlock3DRegionReader::GetMapProjection());
 };

protected:

 virtual const float *GetDataRange() const {
	return(WaveletBlock3DRegionReader::GetDataRange());
 }
 virtual int	OpenVariableReadNative(
	size_t timestep,
	const char *varname,
	int reflevel = 0,
	int lod = 0
 ) {
	return(WaveletBlock3DRegionReader::OpenVariableRead(
		timestep, varname, reflevel)
	); 
 };

 virtual int	CloseVariableNative() {
	 return (WaveletBlock3DRegionReader::CloseVariable());
 };

 virtual int    BlockReadRegionNative(
    const size_t bmin[3], const size_t bmax[3],
    float *region
 )  {
 	return(WaveletBlock3DRegionReader::BlockReadRegion(
		bmin, bmax, region, 1)
	);
 }; 

 virtual void GetValidRegionNative(
    size_t min[3], size_t max[3], int reflevel
 ) const {
 	return(WaveletBlock3DRegionReader::GetValidRegion(
		min, max, reflevel)
	);
 };

 virtual void   GetDimNative(size_t dim[3], int reflevel) const {
	return(WaveletBlock3DRegionReader::GetDim(dim, reflevel));
 };

 virtual void   GetDimBlkNative(size_t bdim[3], int reflevel) const {
	return(WaveletBlock3DRegionReader::GetDimBlk(bdim, reflevel));
 };

 virtual void   MapVoxToUserNative(
    size_t timestep,
    const size_t vcoord0[3], double vcoord1[3], int reflevel = 0
 ) const {
	return(WaveletBlock3DRegionReader::MapVoxToUser(
		timestep, vcoord0, vcoord1, reflevel)
	);
 };

 virtual void   MapUserToVoxNative(
    size_t timestep,
    const double vcoord0[3], size_t vcoord1[3], int reflevel = 0
 ) const {
	return(WaveletBlock3DRegionReader::MapUserToVox(
		timestep, vcoord0, vcoord1, reflevel)
	);
 };

};

};

#endif	//	_DataMgrLayered_h_
