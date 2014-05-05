//
//      $Id: MetadataWRF.h,v 1.6 2011/05/31 21:00:29 alannorton Exp $
//


#ifndef _MetadataWRF_h_
#define _MetadataWRF_h_

#include <vector>
#include <expat.h>
#include <vapor/MyBase.h>
#include <vapor/common.h>
#include <vapor/Metadata.h>
#include <vapor/XmlNode.h>
#include <vapor/ExpatParseMgr.h>
#ifdef WIN32
#pragma warning(disable : 4251)
#endif

namespace VAPoR {

class VDF_API MetadataWRF : public Metadata, public VetsUtil::MyBase {
public:
 MetadataWRF ();
 MetadataWRF (const vector<string> &files);
 MetadataWRF (
	const vector<string> &files, const map <string, string> &atypnames
 );
 ~MetadataWRF();

//! Return the internal blocking factor use for WaveletBlock files
//!
//! \retval size Internal block factor
//! \remarks Required element
//
      virtual const size_t *GetBlockSize() const { return(Dimens); }

//! Return the internal blocking factor at a particular refinement level
//!
//! \param[in] reflevel Refinement level of the variable
//! \param[out] dim Transformed dimension.
//!
//! \retval size Internal block factor
//
      virtual void GetBlockSize(size_t bs[3], int reflevel) const {
        for (int i=0; i<3; i++) bs[i] = Dimens[i]; }

//! Returns the X,Y,Z coordinate dimensions of the data in grid coordinates
//! \retval dim A three element vector containing the voxel dimension of
//! the data at its native resolution
//!
//! \remarks Required element
//
      const size_t *GetDimension() const { return(Dimens); }

      void GetGridDim(size_t dim[3]) const { 
        for (int i=0; i<3; i++) dim[i] = Dimens[i]; 
      }

//! Return the domain extents specified in user coordinates
//!
//! \retval extents A six-element array containing the min and max
//! bounds of the data domain in user-defined coordinates
//!
//! \remarks Required element
//
      vector<double> GetExtents() const {
        return(Extents); };

//! Return the Global Attributes from the WRF file used by Vapor.
//! This is very useful in determining if working with a PlanetWRF
//! dataset or not.
//!
//! \retval Global_attrib a vector with attribute value pairs.
//

      vector<pair<string, double> > GetGlobalAttributes() const {
        return(Global_attrib); };

 //! Return the domain extents specified in user coordinates
 //! for the indicated time step
 //!
 //! \param[in] ts A valid data set time step in the range from zero to
 //! GetNumTimeSteps() - 1.
 //! \retval extents A six-element array containing the min and max
 //! bounds of the data domain in user-defined coordinates.
 //! An empty vector is returned if the extents for the specified time
 //! step is not defined.
 //!
 //! \remarks Optional element
 //
 vector<double> GetTSExtents(size_t ts) const {
	if (ts >= Time_extents.size() ) return(Extents);
	return( Time_extents[ts].second);
 }; 

//! Return the number of time steps in the collection
//!
//! \retval value The number of time steps or a negative number on error
//!
//! \remarks Required element
//
      long GetNumTimeSteps() const {return(Time_latlon_extents.size()); };

//! Return the names of the 3D variables in the collection
//!
//! \retval value is a space-separated list of 3D variable names
//!
//! \remarks Required element (VDF version 1.3 or greater)
//
      vector <string> GetVariables3D() const {
        return(Vars3D); };

//! Return the names of the 2D, XY variables in the collection
//!
//! \retval value is a space-separated list of 2D XY variable names
//!
//! \remarks Required element (VDF version 1.3 or greater)
//
      vector <string> GetVariables2DXY() const {
        return(Vars2Dxy); };
      vector <string> GetVariables2DXZ() const {
        vector <string> svec; svec.clear();
        return(svec); };
      vector <string> GetVariables2DYZ() const {
        vector <string> svec; svec.clear();
        return(svec); };

//! Return the max and values of the Lat and lon.
//!
//! \retval float of value.
//
	float GetminLat() const {return (minLat); } ;
	float GetmaxLat() const {return (maxLat); } ;
	float GetminLon() const {return (minLon); } ;
	float GetmaxLon() const {return (maxLon); } ;

//! Return the map projection argument string, if it exists
//!
//! \retval value The map projection string. An empty string is returned
//! if a map projection is not defined.
//!
//! \remarks Optional element
//!
//

	string GetMapProjection() const {
		return(MapProjection); };

//! Reproject the Lat-Lon values for each time step. 
//!
//! \param[in] MapProcjection is a valid string used by Proj4. 
//! \retval Indicated any errors form Proj4.
//!
//

	int ReprojectTsLatLon(string mapstr);

//! Find the lat/lon extents when using rotated lat long 
//!
//! \param[out] exts is a 4-tuple of extents in degrees
//! \retval Nonzero indicates error from Proj4.
//!
//

	int GetRotatedLatLonExtents(double exts[4]);

//! Return the time for a time step, if it exists,
//!
//! \param[in] ts A valid data set time step in the range from zero to
//! GetNumTimeSteps() - 1.
//! \retval value A single element vector specifying the time
//!
//! \remarks Required element
//!
//
 double GetTSUserTime(size_t ts) const { 
	if (ts >= Time_latlon_extents.size()) return(0.0);
	return(Time_latlon_extents[ts].first); 
 };

 void GetTSUserTimeStamp(size_t ts, string &s) const { 
	s.clear();
	if (ts >= UserTimeStamps.size()) return;
	s = UserTimeStamps[ts]; 
 };

 //! Return the grid type.
 //!
 //! \retval type The grid type
 //!
 //! \remarks Required element
 //
      string GetGridType() const { return("layered"); };

//! Return a three-element integer array indicating the coordinate
//! ordering permutation.
//!
//! \retval integer-vector
//!
//! \remarks Optional element
//
      vector<long> GetGridPermutation() const {
        return(GridPermutation); };

//! Return a three-element boolean array indicating if the X,Y,Z
//! axes have periodic boundaries, respectively.
//!
//! \retval boolean-vector
//!
//! \remarks Required element (VDF version 1.3 or greater)
//
      vector<long> GetPeriodicBoundary() const {
        return(PeriodicBoundary); };

//! Map a VDC time step to a netCDF file and time offset
//!
//! This method maps the global VDC time step, \p timestep, into 
//! the WRF netCDF file name, \p wrf_fname, and a local time offset,
//! \p wrf_ts, within that file. The valid range for \p timestep is
//! 0 to GetNumTimeSteps()-1. 
//!
//! \param[in] timestep A VDC time step
//! \param[out] wrf_fname The WRF netCDF file name
//! \param[out] wrf_ts The integer time offset within the file \p wrf_fname
//!
//! \retval status A negative value is returned if \p timestep is not valid
//! \sa MapWRFTimestep()
//

      int MapVDCTimestep(
        size_t timestep, 
        string &wrf_fname,
        size_t &wrf_ts
      );

//! Map a netCDF file name and time offset to a global VDC time step
//!
//! This method maps the WRF netCDF file name, \p wrf_fname, and a local 
//! time offset, \p wrf_ts, within that file to the global VDC time 
//! step, \p timestep. 
//!
//! \param[in] wrf_fname The WRF netCDF file name
//! \param[in] wrf_ts The integer time offset within the file \p wrf_fname
//! \param[out] timestep A VDC time step
//!
//! \retval status A negative value is returned if \p wrf_fname is
//! unrecognized or \p wrf_ts is not a valid time step within 
//! the file \p wrf_fname.
//!
//! \sa MapVDCTimestep()
//

      int MapWRFTimestep(
        const string &wrf_fname,
        size_t wrf_ts,
        size_t &timestep 
      );

protected:
 map <string, string> GetAtypNames() const {return(_atypnames); };

private:
      vector<pair<string, double> > Global_attrib;
      vector <pair< TIME64_T, vector <float> > > Time_latlon_extents;
      vector<pair<TIME64_T, vector<double> > > Time_extents;
      vector<pair<TIME64_T, string> > Timestep_filename;
      vector<pair<TIME64_T, size_t> > Timestep_offset;
      vector<string> Vars3D, Vars2Dxy, UserTimeStamps;
      vector<long> PeriodicBoundary, GridPermutation;
      vector<double> Extents, UserTimes;
      string StartDate, MapProjection;
      size_t Dimens[3];
      float minLat, minLon, maxLat, maxLon;
      double Reflevel;

 map <string, string> _atypnames;

 void _MetadataWRF (
	const vector<string> &files, const map <string, string> &atypnames
 );


}; // End of class.

}; // End of namespace.

#endif  // _MetadataWRF_h_

