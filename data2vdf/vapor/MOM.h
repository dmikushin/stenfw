//
//      $Id: MOM.h,v 1.1 2011/10/19 16:46:32 alannorton Exp $
//


#ifndef _MOM_h_
#define _MOM_h_


#include <iostream>
#include <vector>
#include <map>
#include <sstream>
#include <netcdf.h>
#include <vapor/PVTime.h>

namespace VAPoR {

class VDF_API MOM : public VetsUtil::MyBase {
public:

 // A struct for storing info about a netCDF variable
 typedef struct {
	string momvname; // Name of the variable as it appears in the MOM file
	string alias; // alias for momvname
	int varid; // Variable ID in netCDF file
	vector <int> dimids; // Array of dimension IDs that this variable uses
	vector <size_t> dimlens; // Array of dimensions from netCDF file
	nc_type xtype; // The type of the variable (float, double, etc.)
	vector <bool> stag; // Indicates which of the fastest varying three dimensions 
				// are staggered. N.B. order of array is reversed from
				// netCDF convection. I.e. stag[0] == X, stag[2] == Z.
 } varInfo_t;


 // File handle
 //
 typedef struct {
	varInfo_t thisVar;
	float *buffer;	// space to buffer a 2D slice of data
	int z;		// Z coordinate of slice in 3D var (0 if 2D)
	int momt;		// time step offset in file
 } varFileHandle_t;

 MOM(const string &toponame, const vector<string>& vars2d, const vector<string>& vars3d);

 MOM(const string &toponame,  const map <string, string> &atypnames, const vector<string>& vars2d, const vector<string>& vars3d);
 virtual ~MOM();

 varFileHandle_t *Open(const string &varname);
 int Close(varFileHandle_t *fh);

 int addFile(const string& momname, float extents[6], vector<string>& vars2d, vector<string>& vars3d);

 // structure for storing dimension info
 typedef struct {
    char name[NC_MAX_NAME+1];	// dim name
    size_t size;				// dim len
 } ncdim_t;


 // Reads a horizontal slice of the variable indicated by thisVar, 
 // and interpolates any points on a staggered grid to an unstaggered grid
 //
 int GetZSlice(
	varFileHandle_t *fh,
	size_t momt,    // MOM time step
	size_t z, // The (unstaggered) vertical coordinate of the plane we want
	float *buffer
 );
 double getStartSeconds(){return startTimeDouble;} //seconds since 1/1/1970, + or -

 
 const vector<double>& GetTimes(){return _momTimes;}

private:

 // A mapping between required MOM variable names and how  these
 // names may appear in the file. The first string is the alias,
 // the second is the name of the var as it appears in the file
 //
 map <string, string> _atypnames;

 int _ncid;
 float _Exts[6]; // extents
 size_t _dimLens[4]; // Lengths of x, y, z, and time dimensions (unstaggered)
 
 bool add2dVars, add3dVars;  //Should we add to the existing variable names?
 vector<double> _momTimes;  //Times for which valid data has been found
 
 vector <string> geolatvars; //up to two geolat var names.  First is T-grid
 vector <string> geolonvars; //up to two geolon var names.  First is T-grid
 
string startTimeStamp; //Start time found in metadata of time variable
double startTimeDouble; // Conversion of start time into seconds since 1/1/1970, or -1.e30 if not present
 
 int _MOM(const string &momname, const map <string, string> &atypnames, const vector<string>& vars2d, const vector<string>& vars3d);

 void _InterpHorizSlice(
	float * fbuffer, // The slice of data to interpolate
	const varInfo_t & thisVar // Data about the variable
 );


 // Reads a single horizontal slice of netCDF data
 int _ReadZSlice4D(
	int ncid, // ID of the netCDF file
	const varInfo_t & thisVar, // Struct for the variable we want
	size_t momT, // The MOM time step we want
	size_t z, // Which z slice we want
	float * fbuffer // Buffer we're going to store slice in
 );

 // Reads a single horizontal slice of netCDF data
 int _ReadZSlice3D(
	int ncid, // ID of the netCDF file
	const varInfo_t & thisVar, // Struct for the variable we want
	size_t momT, // The MOM time step we want
	float * fbuffer // Buffer we're going to store slice in
 );

 int _GetMOMTopo(
    int ncid // Holds netCDF file ID (in)
 );
 void addTimes(int numtimes, double times[]);
 void addVarName(int dim, string& vname, vector<string>&vars2d, vector<string>&vars3d);
 int varIsValid(int ncid, int ndims, int varid);

 //Check for geolat or geolon attributes of a variable.
 //Return 1 for geolat, 2 for geolon, 0 for neither.
int testVarAttribs(int ncid, int varid);

 vector<varInfo_t> _momVarInfo;

 // Gets info about a 2D or 3D variable and stores that info in thisVar.
 int	_GetVarInfo(
	int ncid, // ID of the file we're reading
	string name,	// actual name of variable in MOM file (not atypical name)
	const vector <ncdim_t> &ncdims,
	varInfo_t & thisVar // Variable info
 );


};
};

#endif
