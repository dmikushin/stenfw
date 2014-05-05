//
//      $Id: DataMgrAMR.h,v 1.4 2010/09/24 17:14:07 southwic Exp $
//

#ifndef	_DataMgrAMR_h_
#define	_DataMgrAMR_h_


#include <vector>
#include <string>
#include <vapor/DataMgr.h>
#include <vapor/AMRIO.h>
#include <vapor/common.h>

namespace VAPoR {

//
//! \class DataMgrAMR
//! \brief A cache based data reader
//! \author John Clyne
//! \version $Revision: 1.4 $
//! \date    $Date: 2010/09/24 17:14:07 $
//!
//
class VDF_API DataMgrAMR : public DataMgr, public AMRIO {

public:

 DataMgrAMR(
	const string &metafile,
	size_t mem_size
 );
// ) : DataMgr(mem_size), AMRIO(metafile()) {};

 DataMgrAMR(
	const MetadataVDC &metadata,
	size_t mem_size
 );


 virtual ~DataMgrAMR() {}; 

 virtual int _VariableExists(
	size_t ts,
	const char *varname,
	int reflevel = 0,
	int lod = 0
 ) const {
	return (AMRIO::VariableExists(ts,varname,reflevel));
 };

 //
 //	Metadata methods
 //

 virtual const size_t *GetBlockSize() const {
	return(AMRIO::GetBlockSize());
 }

 virtual void GetBlockSize(size_t bs[3], int reflevel) const {
	AMRIO::GetBlockSize(bs, reflevel);
 }

 virtual int GetNumTransforms() const {
	return(AMRIO::GetNumTransforms());
 };

 virtual vector<double> GetExtents() const {
	return(AMRIO::GetExtents());
 };

 virtual long GetNumTimeSteps() const {
	return(AMRIO::GetNumTimeSteps());
 };

 virtual vector <string> _GetVariables3D() const {
	return(AMRIO::GetVariables3D());
 };

 virtual vector <string> _GetVariables2DXY() const {
	return(AMRIO::GetVariables2DXY());
 };
 virtual vector <string> _GetVariables2DXZ() const {
	return(AMRIO::GetVariables2DXZ());
 };
 virtual vector <string> _GetVariables2DYZ() const {
	return(AMRIO::GetVariables2DYZ());
 };

 virtual vector<long> GetPeriodicBoundary() const {
	return(AMRIO::GetPeriodicBoundary());
 };

 virtual vector<long> GetGridPermutation() const {
	return(AMRIO::GetGridPermutation());
 };

 virtual double GetTSUserTime(size_t ts) const {
	return(AMRIO::GetTSUserTime(ts));
 };

 virtual void GetTSUserTimeStamp(size_t ts, string &s) const {
	AMRIO::GetTSUserTimeStamp(ts,s);
 };

 virtual void   GetGridDim(size_t dim[3]) const {
	return(AMRIO::GetGridDim(dim));
 };

 virtual string GetCoordSystemType() const {
	return(AMRIO::GetCoordSystemType());
 };
	



protected:

 virtual int	OpenVariableRead(
	size_t timestep,
	const char *varname,
	int reflevel = 0,
	int lod = 0
 );

 virtual int	CloseVariable() {
	 return (AMRIO::CloseVariable());
 };

 virtual int    BlockReadRegion(
    const size_t bmin[3], const size_t bmax[3],
    float *region
 ); 

 virtual void GetValidRegion(
    size_t min[3], size_t max[3], int reflevel
 ) const {
 	return(AMRIO::GetValidRegion(
		min, max, reflevel)
	);
 };

 virtual const float *GetDataRange() const {
	return(AMRIO::GetDataRange());
 }

private:
	AMRTree _amrtree;
	int _reflevel;

};

};

#endif	//	_DataMgrAMR_h_
