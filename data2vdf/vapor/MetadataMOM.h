//
//      $Id: MetadataMOM.h,v 1.1 2011/10/19 16:46:32 alannorton Exp $
//


#ifndef _MetadataMOM_h_
#define _MetadataMOM_h_

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

class VDF_API MetadataMOM : public Metadata, public VetsUtil::MyBase {
public:
 MetadataMOM ();
 MetadataMOM (const vector<string> &files, const string& topofile, vector<string>& vars2d, vector<string>& vars3d);
 MetadataMOM (
	const vector<string> &files, const string& topofile, const map <string, string> &atypnames, vector<string>& vars2d, vector<string>& vars3d
 );
 ~MetadataMOM();

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



 

//! Return the number of time steps in the collection
//!
//! \retval value The number of time steps or a negative number on error
//!
//! \remarks Required element
//
      long GetNumTimeSteps() const {return(UserTimes.size()); };

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
		return(string("+proj=latlon +ellps=sphere")); };



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
	if (ts >= UserTimes.size()) return(0.0);
	//Convert days to seconds
	return(UserTimes[ts]); 
 };

 void GetTSUserTimeStamp(size_t ts, string &s) const ;

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



protected:
 map <string, string> GetAtypNames() const {return(_atypnames); };

private:
      
     
      vector<string> Vars3D, Vars2Dxy;
      vector<long> PeriodicBoundary, GridPermutation;
      vector<double> Extents, UserTimes;
	  double epochStartTimeSeconds;
      size_t Dimens[3];
      float minLat, minLon, maxLat, maxLon;
      double Reflevel;

 map <string, string> _atypnames;

 void _MetadataMOM (
	const vector<string> &files, const string& topofile, const map <string, string> &atypnames, vector<string>& vars2d, vector<string>& vars3d
 );


}; // End of class.

}; // End of namespace.

#endif  // _MetadataMOM_h_

