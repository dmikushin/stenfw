//
//      $Id: DataMgrWC.h,v 1.4 2010/09/24 17:14:07 southwic Exp $
//

#ifndef	_DataMgrWC_h_
#define	_DataMgrWC_h_


#include <vector>
#include <string>
#include <vapor/DataMgr.h>
#include <vapor/WaveCodecIO.h>
#include <vapor/common.h>

namespace VAPoR {

//
//! \class DataMgrWC
//! \brief A cache based data reader
//! \author John Clyne
//! \version $Revision: 1.4 $
//! \date    $Date: 2010/09/24 17:14:07 $
//!
//! This class provides a wrapper to the WaveCodecIO()
//! classes that includes a memory cache. Data regions read from disk through 
//! this
//! interface are stored in a cache in main memory, where they may be
//! be available for future access without reading from disk.
//
class VDF_API DataMgrWC : public DataMgr, public WaveCodecIO {

public:

 DataMgrWC(
	const string &metafile,
	size_t mem_size
 );
// ) : DataMgr(mem_size), WaveCodecIO(metafile()) {};

 DataMgrWC(
	const MetadataVDC &metadata,
	size_t mem_size
 );


 virtual ~DataMgrWC() {}; 

 virtual int _VariableExists(
	size_t ts,
	const char *varname,
	int reflevel = 0,
	int lod = 0
 ) const {
	return (WaveCodecIO::VariableExists(ts,varname,reflevel, lod));
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
 };

 virtual void   GetGridDim(size_t dim[3]) const {
	return(WaveCodecIO::GetGridDim(dim));
 };

 virtual string GetMapProjection() const {
	return(WaveCodecIO::GetMapProjection());
 };

 virtual string GetCoordSystemType() const {
	return(WaveCodecIO::GetCoordSystemType());
 };
	



protected:

 virtual int	OpenVariableRead(
	size_t timestep,
	const char *varname,
	int reflevel = 0,
	int lod = 0
 ) {
	return(WaveCodecIO::OpenVariableRead(
		timestep, varname, reflevel, lod)
	); 
 };

 virtual int	CloseVariable() {
	 return (WaveCodecIO::CloseVariable());
 };

 virtual int    BlockReadRegion(
    const size_t bmin[3], const size_t bmax[3],
    float *region
 )  {
 	return(WaveCodecIO::BlockReadRegion(
		bmin, bmax, region, 1)
	);
 }; 

 virtual void GetValidRegion(
    size_t min[3], size_t max[3], int reflevel
 ) const {
 	return(WaveCodecIO::GetValidRegion(
		min, max, reflevel)
	);
 };

 virtual const float *GetDataRange() const {
	return(WaveCodecIO::GetDataRange());
 }

};

};

#endif	//	_DataMgrWC_h_
