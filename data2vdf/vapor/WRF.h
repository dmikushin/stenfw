//
//      $Id: WRF.h,v 1.12 2010/09/22 22:08:48 clynejp Exp $
//


#ifndef _WRF_h_
#define _WRF_h_


#include <iostream>
#include <vector>
#include <map>
#include <sstream>
#include <netcdf.h>
#include <vapor/PVTime.h>

namespace VAPoR {

class VDF_API WRF : public VetsUtil::MyBase {
public:

 // A struct for storing info about a netCDF variable
 typedef struct {
	string wrfvname; // Name of the variable as it appears in the WRF file
	string alias; // alias for wrfvname
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
	int wrft;		// time step offset in file
 } varFileHandle_t;

 WRF(const string &wrfname);

 WRF(const string &wrfname, const map <string, string> &atypnames);
 virtual ~WRF();

 varFileHandle_t *Open(const string &varname);
 int Close(varFileHandle_t *fh);



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
	size_t wrft,    // WRF time step
	size_t z, // The (unstaggered) vertical coordinate of the plane we want
	float *buffer
 );


 static int WRFTimeStrToEpoch(
	const string &wrftime,
	TIME64_T *seconds,
	int dpy = 0
 );

 static int EpochToWRFTimeStr(
	TIME64_T seconds,
	string &wrftime,
	int daysyear = 0
 );

 void GetWRFMeta(
	float * vertExts, // Vertical extents (out)
	size_t dimLens[4], // Lengths of x, y, z, and time unstaggered dimensions (out)
	string &startDate, // Place to put START_DATE attribute (out)
	string &mapProj, // Map projection string (out)
	vector <string> & wrfVars3d, // 3D Variable names in WRF file (out)
	vector <string> & wrfVars2d, // 2D Variable names in WRF file (out)
	vector <pair<string, double> > & gl_attrib,
	vector <pair< TIME64_T, vector <float> > > &tstepExtents // Time stamps, in seconds, and matching extents (out)
);


private:

 // A mapping between required WRF variable names and how  these
 // names may appear in the file. The first string is the alias,
 // the second is the name of the var as it appears in the file
 //
 map <string, string> _atypnames;

 int _ncid;
 float _vertExts[2]; // Vertical extents
 size_t _dimLens[4]; // Lengths of x, y, z, and time dimensions (unstaggered)
 string _startDate; // Place to put START_DATE attribute 
 string _mapProjection; //PROJ4 projection string
 vector <string> _wrfVars3d;
 vector <string> _wrfVars2d;
 vector <pair<string, double> > _gl_attrib;
 vector <pair< TIME64_T, vector <float> > > _tsExtents; //Times in seconds, lat/lon corners


 int _WRF(const string &wrfname, const map <string, string> &atypnames);

 void _InterpHorizSlice(
	float * fbuffer, // The slice of data to interpolate
	const varInfo_t & thisVar // Data about the variable
 );

 //Construct a proj4 projection string from metadata in a WRF file
 int _GetProjectionString(int ncid, string& projString);

 int _GetCornerCoords(
	 int ncid,
	 int ts, //time step in file
	 const varInfo_t &latInfo, 
	 const varInfo_t &lonInfo, 
	 float coords[8]
 );

 // Reads a single horizontal slice of netCDF data
 int _ReadZSlice4D(
	int ncid, // ID of the netCDF file
	const varInfo_t & thisVar, // Struct for the variable we want
	size_t wrfT, // The WRF time step we want
	size_t z, // Which z slice we want
	float * fbuffer // Buffer we're going to store slice in
 );

 // Reads a single horizontal slice of netCDF data
 int _ReadZSlice3D(
	int ncid, // ID of the netCDF file
	const varInfo_t & thisVar, // Struct for the variable we want
	size_t wrfT, // The WRF time step we want
	float * fbuffer // Buffer we're going to store slice in
 );

 int _GetWRFMeta(
    int ncid, // Holds netCDF file ID (in)
    float *vertExts, 
    size_t dimLens[4], 
    string &startDate, 
    string &mapProjection, 
    vector<string> &wrfVars3d,
    vector<string> &wrfVars2d,
    vector<varInfo_t> &wrfVarInfo,
    vector<pair<string, double> > &gl_attrib,
    vector <pair< TIME64_T, vector <float> > > &tsExtents //Times in seconds, lat/lon corners (out)
 );


 vector<varInfo_t> _wrfVarInfo;

 // Gets info about a 2D or 3D variable and stores that info in thisVar.
 int	_GetVarInfo(
	int ncid, // ID of the file we're reading
	string name,	// actual name of variable in WRF file (not atypical name)
	const vector <ncdim_t> &ncdims,
	varInfo_t & thisVar // Variable info
 );


};
};

#endif
