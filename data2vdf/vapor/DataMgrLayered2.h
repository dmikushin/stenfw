//
//      $Id: DataMgrLayered2.h,v 1.1 2011/07/29 14:44:12 clynejp Exp $
//

#ifndef	_DataMgrLayered2_h_
#define	_DataMgrLayered2_h_


#include <vector>
#include <string>
#include <vapor/WaveCodecIO.h>
#include <vapor/LayeredIO.h>
#include <vapor/common.h>

namespace VAPoR {

//
//! \class DataMgrLayered2
//! \brief A cache based data reader
//! \author John Clyne
//! \version $Revision: 1.1 $
//! \date    $Date: 2011/07/29 14:44:12 $
//!
//! This class provides a wrapper to the Layered()
//! and WaveletBlock2DRegionReader()
//! classes that includes a memory cache. Data regions read from disk through 
//! this
//! interface are stored in a cache in main memory, where they may be
//! be available for future access without reading from disk.
//
class VDF_API DataMgrLayered2 : public LayeredIO, public WaveCodecIO {

public:

 DataMgrLayered2(
	const string &metafile,
	size_t mem_size
 );

 DataMgrLayered2(
	const MetadataVDC &metadata,
	size_t mem_size
 );


 virtual ~DataMgrLayered2() {CloseVariableNative();};

 virtual int _VariableExists(
	size_t ts,
	const char *varname,
	int reflevel = 0,
	int lod = 0
 ) const {
	return (WaveCodecIO::VariableExists(ts,varname,reflevel,lod));
 };

 //
 //	Metadata methods
 //

 virtual const size_t *GetBlockSize() const {
	return(MetadataVDC::GetBlockSize());
 }

 virtual void GetBlockSize(size_t bs[3], int reflevel) const {
	WaveCodecIO::GetBlockSize(bs, reflevel);
 }

 virtual int GetNumTransforms() const {
	return(WaveCodecIO::GetNumTransforms());
 };

 virtual vector <size_t> GetCRatios() const {
    return(WaveCodecIO::GetCRatios());
 };

 virtual vector<double> GetExtents() const {
	return(WaveCodecIO::GetExtents());
 };

 virtual long GetNumTimeSteps() const {
	return(WaveCodecIO::GetNumTimeSteps());
 };

 virtual vector <string> _GetVariables3D() const {
	return(WaveCodecIO::GetVariables3D());
 };

 virtual vector <string> _GetVariables2DXY() const {
	return(WaveCodecIO::GetVariables2DXY());
 };
 virtual vector <string> _GetVariables2DXZ() const {
	return(WaveCodecIO::GetVariables2DXZ());
 };
 virtual vector <string> _GetVariables2DYZ() const {
	return(WaveCodecIO::GetVariables2DYZ());
 };

 virtual vector<long> GetPeriodicBoundary() const {
	return(WaveCodecIO::GetPeriodicBoundary());
 };

 virtual vector<long> GetGridPermutation() const {
	return(WaveCodecIO::GetGridPermutation());
 };

 virtual double GetTSUserTime(size_t ts) const {
	return(WaveCodecIO::GetTSUserTime(ts));
 };

 virtual void GetTSUserTimeStamp(size_t ts, string &s) const {
    WaveCodecIO::GetTSUserTimeStamp(ts,s);
 }

 virtual vector<double> GetTSExtents(size_t ts) const {
	return(WaveCodecIO::GetTSExtents(ts));
 };

 virtual string GetMapProjection() const {
	return(WaveCodecIO::GetMapProjection());
 };

protected:

 virtual const float *GetDataRange() const {
	return(WaveCodecIO::GetDataRange());
 }
 virtual int	OpenVariableReadNative(
	size_t timestep,
	const char *varname,
	int reflevel = 0,
	int lod = 0
 ) {
	return(WaveCodecIO::OpenVariableRead(
		timestep, varname, reflevel, lod)
	); 
 };

 virtual int	CloseVariableNative() {
	 return (WaveCodecIO::CloseVariable());
 };

 virtual int    BlockReadRegionNative(
    const size_t bmin[3], const size_t bmax[3],
    float *region
 )  {
 	return(WaveCodecIO::BlockReadRegion(
		bmin, bmax, region, 1)
	);
 }; 

 virtual void GetValidRegionNative(
    size_t min[3], size_t max[3], int reflevel
 ) const {
 	return(WaveCodecIO::GetValidRegion(
		min, max, reflevel)
	);
 };

 virtual void   GetDimNative(size_t dim[3], int reflevel) const {
	return(WaveCodecIO::GetDim(dim, reflevel));
 };

 virtual void   GetDimBlkNative(size_t bdim[3], int reflevel) const {
	return(WaveCodecIO::GetDimBlk(bdim, reflevel));
 };

 virtual void   MapVoxToUserNative(
    size_t timestep,
    const size_t vcoord0[3], double vcoord1[3], int reflevel = 0
 ) const {
	return(WaveCodecIO::MapVoxToUser(
		timestep, vcoord0, vcoord1, reflevel)
	);
 };

 virtual void   MapUserToVoxNative(
    size_t timestep,
    const double vcoord0[3], size_t vcoord1[3], int reflevel = 0
 ) const {
	return(WaveCodecIO::MapUserToVox(
		timestep, vcoord0, vcoord1, reflevel)
	);
 };

};

};

#endif	//	_DataMgrLayered2_h_
