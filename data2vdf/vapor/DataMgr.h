//
//      $Id: DataMgr.h,v 1.30 2010/11/15 20:54:30 clynejp Exp $
//

#ifndef	_DataMgr_h_
#define	_DataMgr_h_

#include <list>
#include <map>
#include <string>
#include <vector>
#include <vapor/MyBase.h>
#include <vapor/BlkMemMgr.h>
#include <vapor/Metadata.h>
#include <vapor/common.h>

namespace VAPoR {
class PipeLine;


//
//! \class DataMgr
//! \brief A cache based data reader
//! \author John Clyne
//! \version $Revision: 1.30 $
//! \date    $Date: 2010/11/15 20:54:30 $

//!
//! The DataMgr class is an abstract class that defines public methods for
//! accessing (reading) 2D and 3D field variables. The class implements a 
//! memory cache to speed data access -- once a variable is read it is
//! stored in cache for subsequent access. The DataMgr class is abstract:
//! it declares a number of protected pure virtual methods that must be
//! implemented by specializations of this class.
//
class VDF_API DataMgr : public Metadata, public VetsUtil::MyBase {

public:

 //! Constructor for the DataMgr class. 
 //!
 //! \param[in] mem_size Size of memory cache to be created, specified 
 //! in MEGABYTES!!
 //!
 //! \note The success or failure of this constructor can be checked
 //! with the GetErrCode() method.
 //! 
 //! \sa GetErrCode(), 
 //
 DataMgr(
	size_t mem_size
 );

 virtual ~DataMgr(); 




 //! Read in and return a subregion from the dataset.
 //!
 //! GetRegion() will first check to see if the requested region resides
 //! in cache. If so, no reads are performed. If the named variable is not
 //! in cache, GetRegion() will next check to see if the variable can
 //! be calculated by recursively executing PipeLine stages 
 //! (\see NewPipeline()). Finally,
 //! if the varible is not the result of PipeLine execution GetRegion()
 //! will attempt to access the varible through methods implemented by
 //! derived classes of the DataMgr class.
 //!
 //! The \p ts, \p varname, and \p level pararmeter tuple identifies 
 //! the time step, variable name, and refinement level, 
 //! respectively, of the requested volume.
 //! The \p min and \p max vectors identify the minium and
 //! maximum extents, in block coordinates, of the subregion of interest. The
 //! minimum valid value of \p min is (0,0,0), the maximum valid value of
 //! \p max is (nbx-1,nby-1,nbz-1), where nbx, nby, and nbz are the block
 //! dimensions
 //! of the volume at the resolution indicated by \p level. I.e.
 //! the coordinates are specified relative to the desired volume
 //! resolution. If the requested region is available, GetRegion() returns 
 //! a pointer to memory containing the region. Subsequent calls to GetRegion()
 //! may invalidate the memory space returned by previous calls unless
 //! the \p lock parameter is set, in which case the array returned by
 //! GetRegion() is locked into memory until freed by a call the
 //! UnlockRegion() method (or the class is destroyed).
 //!
 //! GetRegion will fail if the requested data are not present. The
 //! VariableExists() method may be used to determine if the data
 //! identified by a (resolution,timestep,variable) tupple are
 //! available on disk.
 //!
 //! \note The \p lock parameter increments a counter associated 
 //! with the requested region of memory. The counter is decremented
 //! when UnlockRegion() is invoked.
 //!
 //! \param[in] ts A valid time step from the Metadata object used 
 //! to initialize the class
 //! \param[in] varname A valid variable name 
 //! \param[in] reflevel Refinement level requested
 //! \param[in] lod Level of detail requested
 //! \param[in] min Minimum region bounds in blocks
 //! \param[in] max Maximum region bounds in blocks
 //! \param[in] lock If true, the memory region will be locked into the 
 //! cache (i.e. valid after subsequent GetRegion() calls).
 //! \retval ptr A pointer to a region containing the desired data, or NULL
 //! if the region can not be extracted.
 //! \sa NewPipeline(), GetErrMsg()
 //
 float   *GetRegion(
    size_t ts,
    const char *varname,
    int reflevel,
    int lod,
    const size_t min[3],
    const size_t max[3],
    int lock = 0
 );


 //! Read in, quantize and return a subregion from the multiresolution dataset
 //!
 //! This method is identical to the GetRegion() method except that the
 //! data are returned as quantized, 8-bit unsigned integers. 
 //! Regions with integer data types are created by quantizing
 //! native floating point representations such that floating values
 //! less than or equal to \p range[0] are mapped to min, and values 
 //! greater than or equal to \p range[1] are mapped to max, where "min" and
 //! "max" are the minimum and maximum values that may be represented 
 //! by the integer type. For example, for 8-bit, unsigned ints min is 0
 //! and max is 255. Floating point values between \p range[0] and \p range[1]
 //! are linearly interpolated between min and max.
 //!
 //! \param[in] ts A valid time step from the Metadata object used 
 //! to initialize the class
 //! \param[in] varname A valid variable name 
 //! \param[in] reflevel Refinement level requested
 //! \param[in] lod Level of detail requested
 //! \param[in] min Minimum region bounds in blocks
 //! \param[in] max Maximum region bounds in blocks
 //! \param[in] range A two-element vector specifying the minimum and maximum
 //! quantization mapping. 
 //! \param[in] lock If true, the memory region will be locked into the 
 //! \retval ptr A pointer to a region containing the desired data, 
 //! quantized to 8 bits, or NULL
 //! if the region can not be extracted.
 //! \sa GetErrMsg(), GetRegion()
 //
 unsigned char   *GetRegionUInt8(
    size_t ts,
    const char *varname,
    int reflevel,
    int lod,
    const size_t min[3],
    const size_t max[3],
	const float range[2],
    int lock = 0
);

 //! Read in, quantize and return a pair of subregions from the 
 //! multiresolution dataset
 //!
 //! This method is identical to the GetRegionUInt8() method except that the
 //! two variables are read and their values are stored in a single,
 //! interleaved array.
 //!
 //! \param[in] ts A valid time step from the Metadata object used 
 //! to initialize the class
 //! \param[in] varname1 First variable name 
 //! \param[in] varname2 Second variable name 
 //! \param[in] reflevel Refinement level requested
 //! \param[in] lod Level of detail requested
 //! \param[in] min Minimum region bounds in blocks
 //! \param[in] max Maximum region bounds in blocks
 //! \param[in] range1 First variable data range
 //! \param[in] range2 Second variable data range
 //! quantization mapping. 
 //! \param[in] lock If true, the memory region will be locked into the 
 //! \retval ptr A pointer to a region containing the desired data, 
 //! quantized to 8 bits, or NULL
 //! if the region can not be extracted.
 //! \sa GetErrMsg(), GetRegion()
 //
 unsigned char   *GetRegionUInt8(
    size_t ts,
    const char *varname1,
    const char *varname2,
    int reflevel,
    int lod,
    const size_t min[3],
    const size_t max[3],
	const float range1[2],
	const float range2[2],
    int lock = 0
);

 //! Read in, quantize and return a pair of subregions from the 
 //! multiresolution dataset
 //!
 //! This method is identical to the GetRegionUInt16() method except that the
 //! two variables are read and their values are stored in a single,
 //! interleaved array.
 //!
 //! \param[in] ts A valid time step from the Metadata object used 
 //! to initialize the class
 //! \param[in] varname1 First variable name 
 //! \param[in] varname2 Second variable name 
 //! \param[in] reflevel Refinement level requested
 //! \param[in] lod Level of detail requested
 //! \param[in] min Minimum region bounds in blocks
 //! \param[in] max Maximum region bounds in blocks
 //! \param[in] range1 First variable data range
 //! \param[in] range2 Second variable data range
 //! quantization mapping. 
 //! \param[in] lock If true, the memory region will be locked into the 
 //! \retval ptr A pointer to a region containing the desired data, 
 //! quantized to 16 bits, or NULL
 //! if the region can not be extracted.
 //! \sa GetErrMsg(), GetRegion()
 //
 unsigned char   *GetRegionUInt16(
    size_t ts,
    const char *varname1,
    const char *varname2,
    int reflevel,
    int lod,
    const size_t min[3],
    const size_t max[3],
	const float range1[2],
	const float range2[2],
    int lock = 0
);


 //! Read in, quantize and return a subregion from the multiresolution dataset
 //!
 //! This method is identical to the GetRegion() method except that the
 //! data are returned as quantized, 16-bit unsigned integers. 
 //! Regions with integer data types are created by quantizing
 //! native floating point representations such that floating values
 //! less than or equal to \p range[0] are mapped to min, and values 
 //! greater than or equal to \p range[1] are mapped to max, where "min" and
 //! "max" are the minimum and maximum values that may be represented 
 //! by the integer type. For example, for 16-bit, unsigned ints min is 0
 //! and max is 65535. Floating point values between \p range[0] and \p range[1]
 //! are linearly interpolated between min and max.
 //!
 //! \param[in] ts A valid time step from the Metadata object used 
 //! to initialize the class
 //! \param[in] varname A valid variable name 
 //! \param[in] reflevel Refinement level requested
 //! \param[in] lod Level of detail requested
 //! \param[in] min Minimum region bounds in blocks
 //! \param[in] max Maximum region bounds in blocks
 //! \param[in] range A two-element vector specifying the minimum and maximum
 //! quantization mapping. 
 //! \param[in] lock If true, the memory region will be locked into the 
 //! \retval ptr A pointer to a region containing the desired data, 
 //! quantized to 16 bits, or NULL
 //! if the region can not be extracted.
 //! \sa GetErrMsg(), GetRegion()
 //
 unsigned char   *GetRegionUInt16(
    size_t ts,
    const char *varname,
    int reflevel,
    int lod,
    const size_t min[3],
    const size_t max[3],
	const float range[2],
    int lock = 0
);


 //! Unlock a floating-point region of memory 
 //!
 //! Decrement the lock counter associatd with a 
 //! region of memory, and if zero,
 //! unlock region of memory previously locked GetRegion(). 
 //! When the lock counter reaches zero the region is simply 
 //! marked available for
 //! internal garbage collection during subsequent GetRegion() calls
 //!
 //! \param[in] region A pointer to a region of memory previosly 
 //! returned by GetRegion()
 //! \retval status Returns a non-negative value on success
 //!
 //! \sa GetRegion(), GetRegion()
 //
 int	UnlockRegion (
    const void *region
 );

 //! Return the current data range as a two-element array
 //!
 //! This method returns the minimum and maximum data values
 //! for the indicated time step and variable
 //!
 //! \param[in] ts A valid time step from the Metadata object used 
 //! to initialize the class
 //! \param[in] varname Name of variable 
 //! \param[in] reflevel Refinement level requested
 //! \param[in] lod Level of detail requested
 //! \param[out] range  A two-element vector containing the current 
 //! minimum and maximum.
 //! \retval status Returns a non-negative value on success 
 //! quantization mapping.
 //!
 //
 int GetDataRange(
	size_t ts, const char *varname, float range[2],
	int reflevel = 0, int lod = 0
 );

 //! Return the valid region bounds for the specified region
 //!
 //! This method returns the minimum and maximum valid coordinate
 //! bounds (in voxels) of the subregion indicated by the timestep
 //! \p ts, variable name \p varname, and refinement level \p reflevel
 //! for the indicated time step and variable. Data are guaranteed to
 //! be available for this region.
 //!
 //!
 //! \param[in] ts A valid time step from the Metadata object used 
 //! to initialize the class
 //! \param[in] varname Name of variable 
 //! \param[in] reflevel Refinement level of the variable
 //! \param[out] min Minimum coordinate bounds (in voxels) of volume
 //! \param[out] max Maximum coordinate bounds (in voxels) of volume
 //! \retval status A non-negative int is returned on success
 //!
 //
 int GetValidRegion(
    size_t ts,
    const char *varname,
    int reflevel,
    size_t min[3],
    size_t max[3]
 );

 //! Clear the memory cache
 //!
 //! This method clears the internal memory cache of all entries
 //
 void	Clear();

 //! Returns true if indicated data volume is available
 //!
 //! Returns true if the variable identified by the timestep, variable
 //! name, refinement level, and level-of-detail is present in 
 //! the data set. Returns 0 if
 //! the variable is not present.
 //! \param[in] ts A valid time step from the Metadata object used
 //! to initialize the class
 //! \param[in] varname A valid variable name
 //! \param[in] reflevel Refinement level requested. The coarsest 
 //! refinement level is 0 (zero). A value of -1 indicates the finest
 //! refinement level contained in the VDC.
 //! \param[in] lod Compression level of detail requested. The coarsest 
 //! approximation level is 0 (zero). A value of -1 indicates the finest
 //! refinement level contained in the VDC.
 //
 virtual int VariableExists(
	size_t ts,
	const char *varname,
	int reflevel = 0,
	int lod = 0
 ) const;

 //!
 //! Add a pipeline stage to produce derived variables
 //!
 //! Add a new pipline stage for derived variable calculation. If a 
 //! pipeline already exists with the same
 //! name it is replaced. The output variable names are added to
 //! the list of variables available for this data 
 //! set (see GetVariables3D, etc.).
 //!
 //! An error occurs if:
 //!
 //! \li The output variable names match any of the native variable 
 //! names - variable names returned via _GetVariables3D(), etc. 
 //! \li The output variable names match the output variable names
 //! of pipeline stage previously added with NewPipeline()
 //! \li A circular dependency is introduced by adding \p pipeline
 //!
 //! \retval status A negative int is returned on failure.
 //!
 int NewPipeline(PipeLine *pipeline);

 //!
 //! Remove the named pipline if it exists. Otherwise this method is a
 //! no-op
 //!
 //! \param[in] name The name of the pipeline as returned by 
 //! PipeLine::GetName()
 //!
 void RemovePipeline(string name);

 //! \copydoc Metadata::GetVariables3D()
 //
 virtual vector <string> GetVariables3D() const;

 //! \copydoc Metadata::GetVariables2DXY()
 //
 virtual vector <string> GetVariables2DXY() const;

 //! \copydoc Metadata::GetVariables2DXZ()
 //
 virtual vector <string> GetVariables2DXZ() const;

 //! \copydoc Metadata::GetVariables2DYZ()
 //
 virtual vector <string> GetVariables2DYZ() const;

 //! Return true if the named variable is the output of a pipeline
 //!
 //! This method returns true if \p varname matches a variable name
 //! in the output list (PipeLine::GetOutputs()) of any pipeline added
 //! with NewPipeline()
 //!
 //! \sa NewPipeline()
 //
 bool IsVariableDerived(string varname) const;

 //! Return true if the named variable is availble from the derived 
 //! classes data access methods. 
 //!
 //! A return value of true does not imply that the variable can
 //! be read (\see VariableExists()), only that it is part of the 
 //! data set known to the derived class
 //!
 //! \sa NewPipeline()
 //
 bool IsVariableNative(string varname) const;

 virtual VarType_T GetVarType(const string &varname) const; 

	
//! Purge the cache of a variable
//!
//! \param[in] varname is the variable name
//!
void PurgeVariable(string varname);


 enum _dataTypes_t {UINT8,UINT16,UINT32,FLOAT32};
 void    *alloc_region(
	size_t ts,
	const char *varname,
	VarType_T vtype,
	int reflevel,
	int lod,
	_dataTypes_t type,
	const size_t min[3],
	const size_t max[3],
	int lock,
	bool fill
 ); 

protected:
 const vector<string> emptyVec;
 

 // The protected methods below are pure virtual and must be implemented by any 
 // child class  of the DataMgr.


 //! Open the named variable for reading
 //!
 //! This method prepares the multi-resolution, multi-lod data volume, 
 //! indicated by a
 //! variable name and time step pair, for subsequent read operations by
 //! methods of this class.  Furthermore, the number of the refinement level
 //! parameter, \p reflevel indicates the resolution of the volume in
 //! the multiresolution hierarchy, and the \p lod parameter indicates
 //! the level of detail. 
 //!
 //! The valid range of values for
 //! \p reflevel is [0..max_refinement], where \p max_refinement is the
 //! maximum finement level of the data set: Metadata::GetNumTransforms().
 //! A value of zero indicates the
 //! coarsest resolution data, a value of \p max_refinement indicates the
 //! finest resolution data.
 //!
 //! The valid range of values for
 //! \p lod is [0..max_lod], where \p max_lod is the
 //! maximum lod of the data set: Metadata::GetCRatios().size() - 1.
 //! A value of zero indicates the
 //! highest compression ratio, a value of \p max_lod indicates the
 //! lowest compression ratio.
 //!
 //! An error occurs, indicated by a negative return value, if the
 //! volume identified by the {varname, timestep, reflevel, lod} tupple
 //! is not present on disk. Note the presence of a volume can be tested
 //! for with the VariableExists() method.
 //! \param[in] timestep Time step of the variable to read
 //! \param[in] varname Name of the variable to read
 //! \param[in] reflevel Refinement level of the variable. A value of -1
 //! indicates the maximum refinment level defined for the VDC
 //! \param[in] lod Level of detail requested. A value of -1
 //! indicates the lowest compression level available for the VDC
 //!
 //! \sa Metadata::GetVariableNames(), Metadata::GetNumTransforms()
 //!
 virtual int	OpenVariableRead(
	size_t timestep,
	const char *varname,
	int reflevel = 0,
	int lod = 0
 ) = 0;

 //! Close the currently opened variable.
 //!
 //! \sa OpenVariableRead()
 //
 virtual int	CloseVariable() = 0;

 //! Read in and return a subregion from the currently opened multiresolution
 //! data volume.
 //!
 //! The dimensions of the region are provided in block coordinates. However,
 //! the returned region is not blocked. 
 //! 
 //!
 //! \param[in] bmin Minimum region extents in block coordinates
 //! \param[in] bmax Maximum region extents in block coordinates
 //! \param[out] region The requested volume subregion
 //!
 //! \retval status Returns a non-negative value on success
 //! \sa OpenVariableRead(), Metadata::GetBlockSize(), MapVoxToBlk()
 //
 virtual int    BlockReadRegion(
    const size_t bmin[3], const size_t bmax[3],
    float *region
 ) = 0;



 //! Return the valid bounds of the currently opened region
 //!
 //! The data model permits the storage of volume subregions. This method may
 //! be used to query the valid domain of the currently opened volume. Results
 //! are returned in voxel coordinates, relative to the refinement level
 //! indicated by \p reflevel.
 //!
 //!
 //! \param[out] min A pointer to the minimum bounds of the subvolume
 //! \param[out] max A pointer to the maximum bounds of the subvolume
 //! \param[in] reflevel Refinement level of the variable. A value of -1
 //! indicates the maximum refinment level defined for the VDC
 //!
 //! \retval status Returns a negative value if the volume is not opened
 //! for reading.
 //!
 //! \sa OpenVariableWrite(), OpenVariableRead()
 //
 virtual void GetValidRegion(
    size_t min[3], size_t max[3], int reflevel
 ) const = 0;

 //! Return the data range for the currently open variable
 //!
 //! The method returns the minimum and maximum data values, respectively, 
 //! for the variable currently opened. If the variable is not opened,
 //! or if it is opened for writing, the results are undefined.
 //!
 //! \return range A pointer to a two-element array containing the
 //! Min and Max data values, respectively. If the derived class' 
 //! implementation of this method returns NULL, the DataMgr class
 //! will compute the min and max itself.
 //!
 virtual const float *GetDataRange() const = 0;



 //! \copydoc Metadata::VariableExists()
 //
 virtual int _VariableExists(
	size_t ts,
	const char *varname,
	int reflevel = 0,
	int lod = 0
 ) const = 0;

 //! \copydoc Metadata::_GetVariables3D()
 //
 virtual vector <string> _GetVariables3D() const = 0;

 //! \copydoc Metadata::_GetVariables2DXY()
 //
 virtual vector <string> _GetVariables2DXY() const = 0;

 //! \copydoc Metadata::_GetVariables2DXZ()
 //
 virtual vector <string> _GetVariables2DXZ() const = 0;

 //! \copydoc Metadata::_GetVariables2DYZ()
 //
 virtual vector <string> _GetVariables2DYZ() const = 0;



private:



 size_t _mem_size;


 typedef struct {
	size_t ts;
	string varname;
	int reflevel;
	int lod;
	size_t min[3];
	size_t max[3];
	_dataTypes_t	type;
	int lock_counter;
	void *blks;
 } region_t;

 // a list of all allocated regions
 list <region_t> _regionsList;

 // min and max bounds for quantization
 map <string, float *> _quantizationRangeMap;	

 map <size_t, map<string, float> > _dataRangeMinMap;
 map <size_t, map<string, float> > _dataRangeMaxMap;
 map <size_t, map<string, map<int, vector <size_t> > > > _validRegMinMaxMap;

 int	_timestamp;	// access time of most recently accessed region

 BlkMemMgr	*_blk_mem_mgr;

 vector <PipeLine *> _PipeLines;

 void	*get_region_from_cache(
	size_t ts,
	const char *varname,
	int reflevel,
	int lod,
	_dataTypes_t    type,
	const size_t min[3],
	const size_t max[3],
	int lock
 );


 void	free_region(
	size_t ts,
	const char *varname,
	int reflevel,
	int lod,
	_dataTypes_t type,
	const size_t min[3],
	const size_t max[3]
 );

 int	set_quantization_range(const char *varname, const float range[2]);

 void   setDefaultHighLowVals();
 void	free_var(const string &, int do_native);

 int	free_lru();

 int	_DataMgr(size_t mem_size);

 int get_cached_data_range(size_t ts, const char *varname, float range[2]);

 vector <size_t> get_cached_reg_min_max(
	size_t ts, const char *varname, int reflevel
 );

 unsigned char   *get_quantized_region(
	size_t ts, const char *varname, int reflevel, int lod, const size_t min[3],
	const size_t max[3], const float range[2], int lock,
	_dataTypes_t type
 );

 void	quantize_region_uint8(
    const float *fptr, unsigned char *ucptr, size_t size, const float range[2]
 );

 void	quantize_region_uint16(
    const float *fptr, unsigned char *ucptr, size_t size, const float range[2]
 );

 float *execute_pipeline(
	size_t ts, string varname, int reflevel, int lod,
	const size_t min[3], const size_t max[3], int lock
 ); 

 // Check for circular dependencies in a pipeline 
 //
 bool cycle_check(
	const map <string, vector <string> > &graph,
	const string &node,
	const vector <string> &depends
 ) const;

 // Return true if the inputs of a require the outputs of b
 //
 bool depends_on(
	const PipeLine *a, const PipeLine *b
 ) const;

 vector <string> get_native_variables() const;
 vector <string> get_derived_variables() const;

 PipeLine *get_pipeline_for_var(string varname) const;

};

//! \class PipeLine
//!
//! The PipeLine abstract class declares pure virtual methods
//! that may be defined to allow the construction of a data 
//! transformation pipeline. 
//!
class VDF_API PipeLine {
    public:

	//! PipeLine stage constructor
	//!
	//! \param[in] name an identifier
	//! \param[in] inputs A list of variable names required as inputs
	//! \param[out] outputs A list of variable names and variable
	//! type pairs that will be output by the Calculate() method.
	//!
	PipeLine(
		string name,
		vector <string> inputs, 
		vector <pair <string, Metadata::VarType_T> > outputs
	) {
		_name = name;
		_inputs = inputs;
		_outputs = outputs;
	}
	virtual ~PipeLine(){
	}

	//! Execute the pipeline stage
	//!
	//! This pure virtual method is called from the DataMgr whenever 
	//! a variable 
	//! is requested whose name matches one of the output variable
	//! names. All output variables computed - including the requested
	//! one - will be stored in the cache for subsequent retrieval
	//
	virtual int Calculate (
		vector <const float *> input_blks,
		vector <float *> output_blks,	// space for the output variables
		size_t ts, // current time step
		int reflevel, // refinement level
		int lod, //
		const size_t bs[3], // block dimensions
		const size_t min[3],	// dimensions of all variables (in blocks)
		const size_t max[3]
	) = 0;

	//! Returns the PipeLine stages name
	//
	const string &GetName() const {return (_name); };

	//! Returns the PipeLine inputs
	//
	const vector <string> &GetInputs() const {return (_inputs); };

	//! Returns the PipeLine outputs
	//
	const vector <pair <string, Metadata::VarType_T> > &GetOutputs() const { 
		return (_outputs); 
	};
    private:
	string _name;
	vector <string> _inputs;
	vector<pair<string, Metadata::VarType_T> > _outputs;
    };


};

#endif	//	_DataMgr_h_
