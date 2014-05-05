//
//      $Id: DataMgrWRF.h,v 1.3 2010/09/24 17:14:07 southwic Exp $
//

#ifndef	_DataMgrWRF_h_
#define	_DataMgrWRF_h_


#include <vector>
#include <string>
#include <vapor/LayeredIO.h>
#include <vapor/WRFReader.h>
#include <vapor/common.h>

namespace VAPoR {

//
//! \class DataMgrWRF
//! \brief A cache based data reader
//! \author John Clyne
//! \version $Revision: 1.3 $
//! \date    $Date: 2010/09/24 17:14:07 $
//!
//
class VDF_API DataMgrWRF : public LayeredIO, WRFReader {

public:

 DataMgrWRF(
	const vector <string> &files,
	size_t mem_size
 );

 DataMgrWRF(
	const MetadataWRF &metadata,
	size_t mem_size
 );


 virtual ~DataMgrWRF() { CloseVariableNative(); }; 

 virtual int _VariableExists(
	size_t ts,
	const char *varname,
	int reflevel = 0,
	int lod  = 0
 ) const {
	return (WRFReader::VariableExists(ts,varname));
 };

 //
 //	Metadata methods
 //

 virtual const size_t *GetBlockSize() const {
	return(WRFReader::GetBlockSize());
 }

 virtual void GetBlockSize(size_t bs[3], int reflevel) const {
	WRFReader::GetBlockSize(bs, reflevel);
 }

 virtual int GetNumTransforms() const {
	return(WRFReader::GetNumTransforms());
 };

 virtual vector<double> GetExtents() const {
	return(WRFReader::GetExtents());
 };

 virtual long GetNumTimeSteps() const {
	return(WRFReader::GetNumTimeSteps());
 };

 virtual vector <string> _GetVariables3D() const {
	return(WRFReader::GetVariables3D());
 };

 virtual vector <string> _GetVariables2DXY() const {
	return(WRFReader::GetVariables2DXY());
 };
 virtual vector <string> _GetVariables2DXZ() const {
	return(WRFReader::GetVariables2DXZ());
 };
 virtual vector <string> _GetVariables2DYZ() const {
	return(WRFReader::GetVariables2DYZ());
 };

 virtual vector<long> GetPeriodicBoundary() const {
	return(WRFReader::GetPeriodicBoundary());
 };

 virtual vector<long> GetGridPermutation() const {
	return(WRFReader::GetGridPermutation());
 };

 virtual double GetTSUserTime(size_t ts) const {
	return(WRFReader::GetTSUserTime(ts));
 };

 virtual void GetTSUserTimeStamp(size_t ts, string &s) const {
    WRFReader::GetTSUserTimeStamp(ts,s);
 }

 virtual vector<double> GetTSExtents(size_t ts) const {
	return(WRFReader::GetTSExtents(ts));
 };

 virtual string GetMapProjection() const {
	return(WRFReader::GetMapProjection());
 };

	
protected:

 virtual const float *GetDataRange() const {
	return(NULL);	// Not implemented. Let DataMgr figure it out
 }
 virtual int	OpenVariableReadNative(
	size_t timestep,
	const char *varname,
	int,
	int
 ) {
	return(WRFReader::OpenVariableRead(
		timestep, varname)
	); 
 };

 virtual int	CloseVariableNative() {
	 return (WRFReader::CloseVariable());
 };

 virtual int    BlockReadRegionNative(
    const size_t* /* bmin */, const size_t* /* bmax */,
    float *region
 )  {
 	return(WRFReader::ReadVariable(region)
	);
 }; 

 virtual void GetValidRegionNative(
    size_t min[3], size_t max[3], int reflevel
 ) const;

 virtual void   GetDimNative(size_t dim[3], int reflevel) const {
	return(WRFReader::GetDim(dim, reflevel));
 };

 virtual void   GetDimBlkNative(size_t bdim[3], int reflevel) const {
	return(WRFReader::GetDimBlk(bdim, reflevel));
 };

 virtual void   MapVoxToUserNative(
    size_t timestep,
    const size_t vcoord0[3], double vcoord1[3], int reflevel = 0
 ) const {
	return(WRFReader::MapVoxToUser(
		timestep, vcoord0, vcoord1, reflevel)
	);
 };

 virtual void   MapUserToVoxNative(
    size_t timestep,
    const double vcoord0[3], size_t vcoord1[3], int reflevel = 0
 ) const {
	return(WRFReader::MapUserToVox(
		timestep, vcoord0, vcoord1, reflevel)
	);
 };

};

};

#endif	//	_DataMgrWRF_h_
