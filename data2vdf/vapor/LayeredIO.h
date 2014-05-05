//
//      $Id: LayeredIO.h,v 1.14 2011/07/29 14:44:13 clynejp Exp $
//

#ifndef	_LayeredIO_h_
#define	_LayeredIO_h_


#include <vector>
#include <string>
#include <vapor/DataMgr.h>

namespace VAPoR {

//
//! \class LayeredIO
//! \brief A cache based data reader
//! \author John Clyne
//! \version $Revision: 1.14 $
//! \date    $Date: 2011/07/29 14:44:13 $
//!
//! This class provides a wrapper to the Layered()
//! and WaveletBlock2DRegionReader()
//! classes that includes a memory cache. Data regions read from disk through 
//! this
//! interface are stored in a cache in main memory, where they may be
//! be available for future access without reading from disk.
//
class VDF_API LayeredIO : public DataMgr {

public:

 LayeredIO(
	size_t mem_size
 );

 virtual ~LayeredIO(); 

 //
 //	Metadata methods
 //
 virtual void   GetGridDim(size_t dim[3]) const;
 virtual void   GetDim(size_t dim[3], int reflevel) const;
 virtual void   GetDimBlk(size_t bdim[3], int reflevel) const;

 void SetLowVals(const vector<string>& varNames, const vector<float>& values);
 void SetHighVals(const vector<string>& varNames, const vector<float>& values);
 void GetLowVals(vector<string>&varNames, vector<float>&vals);
 void GetHighVals(vector<string>&varNames, vector<float>&vals);

 void SetInterpolateOnOff(bool on) { _interpolateOn = on; };

 int SetGridHeight(size_t height);
 size_t GetGridHeight() const {return _gridHeight;}

 virtual void   MapVoxToUser(
    size_t timestep,
    const size_t vcoord0[3], double vcoord1[3], int reflevel = 0
 ) const;

 virtual void   MapUserToVox(
    size_t timestep,
    const double vcoord0[3], size_t vcoord1[3], int reflevel = 0
 ) const ;


protected:

 virtual int	OpenVariableRead(
	size_t timestep,
	const char *varname,
	int reflevel = 0,
	int lod = 0
 );

 virtual int	CloseVariable();

 virtual int    BlockReadRegion(
    const size_t bmin[3], const size_t bmax[3],
    float *region
 ); 

 virtual void GetValidRegion(
    size_t min[3], size_t max[3], int reflevel
 ) const ;


 //
 // Derived classes must implement these pure virtual methods
 //
 virtual int	OpenVariableReadNative(
	size_t timestep,
	const char *varname,
	int reflevel = 0,
	int lod = 0
 ) = 0;

 virtual int	CloseVariableNative() = 0;

 virtual int    BlockReadRegionNative(
    const size_t bmin[3], const size_t bmax[3],
    float *region
 ) = 0; 

 virtual void GetValidRegionNative(
    size_t min[3], size_t max[3], int reflevel
 ) const = 0;

 virtual void   GetDimNative(size_t dim[3], int reflevel) const = 0;
 virtual void   GetDimBlkNative(size_t bdim[3], int reflevel) const = 0;

 virtual void   MapVoxToUserNative(
    size_t timestep,
    const size_t vcoord0[3], double vcoord1[3], int reflevel = 0
 ) const = 0;

 virtual void   MapUserToVoxNative(
    size_t timestep,
    const double vcoord0[3], size_t vcoord1[3], int reflevel = 0
 ) const = 0;

private:
 bool _interpolateOn;	// Is interpolation on?
 size_t _gridHeight;	// Interpolation grid height

 float *_elevBlkBuf;	// buffer for layered elevation data
 float *_varBlkBuf;	// buffer for layered variable data
 size_t _blkBufSize;	// Size of space allocated to daa buffers


 map <string, float> _lowValMap;
 map <string, float> _highValMap;


 size_t _cacheTimeStep;	// Cached elevation data 
 int _cacheReflevel;	
 int _cacheLOD;	
 size_t _cacheBMin[3];
 size_t _cacheBMax[3];

 //
 // Attributes of currently opened variable
 //
 VarType_T _vtype;	
 int _reflevel;
 int _lod;
 size_t _timeStep;
 string _varName;

 bool _cacheEmpty;
 bool cache_check(
	size_t timestep,
	int reflevel,
	int lod,
	const size_t bmin[3],
	const size_t bmax[3]
 );

 void cache_set(
	size_t timestep,
	int reflevel,
	int lod,
	const size_t bmin[3],
	const size_t bmax[3]
 );

 void cache_clear();

 void _setDefaultHighLowVals();


 int	_LayeredIO();

void _interpolateRegion(
    float *region,			// Destination interpolated ROI
	const float *elevBlks,	// Source elevation ROI
	const float *varBlks,	// Source variable ROI
	const size_t blkMin[3],
	const size_t blkMax[3],	// coords of native (layered grid)
							// ROI specified in blocks 
    size_t zmini,
	size_t zmaxi,			// Z coords extents of interpolated ROI
	float *lowVal,
	float *highVal
) const;

};

};

#endif	//	_LayeredIO_h_
