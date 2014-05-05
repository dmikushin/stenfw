//
// $Id: GetAppPath.h,v 1.3 2011/07/05 22:09:02 clynejp Exp $
//
#ifndef	_GetAppPath_h_
#define	_GetAppPath_h_
#include <vapor/MyBase.h>
#include <vapor/Version.h>

namespace VetsUtil {

PARAMS_API std::string GetAppPath(
	const string &app, const string &name, const vector <string> &paths);

};

#endif
