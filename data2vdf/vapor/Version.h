//
//      $Id: Version.h,v 1.32 2011/04/04 22:07:32 alannorton Exp $
//
//	WARNING: Version.h is generated automatically from Version.h.sed
//
//************************************************************************
//								*
//		     Copyright (C)  2004			*
//     University Corporation for Atmospheric Research		*
//		     All Rights Reserved			*
//								*
//************************************************************************/
//
//	File:		
//
//	Author:		John Clyne
//			National Center for Atmospheric Research
//			PO 3000, Boulder, Colorado
//
//	Date:		Tue Jul 11 11:59:53 MDT 2006
//
//	Description:	Version information.
//

//! \class Version
//! \brief Return version information
//! \author John Clyne
//! \date    Tue Jul 11 11:59:43 MDT 2006
//! A collection of general purpose utilities - things that
//!                  probably should be in the STL but aren't.
//!


#ifndef	_Version_h_
#define	_Version_h_

#include <string>
#include <cstdlib>
#include <vapor/MyBase.h>
#include <vapor/common.h>

using namespace std;

namespace VetsUtil {

//
class COMMON_API Version : public MyBase {

public:
 //! Return the major version number
 //
 static int GetMajor() { return(_majorVersion); }

 //! Return the minor version number
 //
 static int GetMinor() { return(_minorVersion); }

 //! Return the sub minor version number
 //
 static int GetMinorMinor() { return(_minorMinorVersion); }

 //! Return the canonical version number as a formatted string
 //!
 //! Return the canonical version number as a formatted string of
 //! the form: X.Y.Z, where \p X is the major version number, \p Y
 //! is the minor version number, and \p Z is the sub minor version number.
 //
 static const string &GetVersionString();

 //! Return a string containing the date  associated with the version number
 //!
 //! This method returns the value of the RCS \p Date keyword. In general,
 //! this should corespond to the date that the version number was last 
 //! advanced.
 //
 static const string &GetDateString() {
	_dateString.assign("$Date: 2011/04/04 22:07:32 $"); return(_dateString);
 }
 static int Compare(int major, int minor, int minorminor) ;
 static int Compare(std::string ver1, std::string ver2);

private:
 static const int _majorVersion = 2;
 static const int _minorVersion = 1;
 static const int _minorMinorVersion = 0;
 static string _formatString;
 static string _dateString;


};
}

#endif
