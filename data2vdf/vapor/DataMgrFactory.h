//
//      $Id: DataMgrFactory.h,v 1.2 2010/08/26 19:54:31 clynejp Exp $
//

#ifndef	_DataMgrFactory_h_
#define	_DataMgrFactory_h_


#include <vector>
#include <string>
#include <vapor/DataMgr.h>

namespace VAPoR {

class VDF_API DataMgrFactory : public VetsUtil::MyBase { 
public:

 static DataMgr *New(
	const vector <string> &files, size_t mem_size, string ftype = "vdf"
 );

};

};

#endif	//	_DataMgrFactory_h_
