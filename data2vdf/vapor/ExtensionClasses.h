//This header file tells where to find all extension class definitions.
//All such extension classes must be derived from ParamsBase
//These can include Params classes, as well as components such as TransferFunctions
//
//The InstallExtension method installs the extension classes and  (ivoked by VizWinMgr).
#include "../apps/vaporgui/guis/arroweventrouter.h"
#include "../lib/params/arrowparams.h"
#include "../apps/vaporgui/images/arrowrake.xpm"
#ifdef MODELS
#include "../apps/vaporgui/guis/ModelEventRouter.h"
#endif
#include "../lib/params/ModelParams.h"
#include "../lib/params/Transform3d.h"

namespace VAPoR {
	//For each extension class, insert the methods ParamsBase::RegisterParamsBaseClass and VizWinMgr::InstallTab
	//Into the following method:
	static void InstallExtensions(){
           ParamsBase::RegisterParamsBaseClass(ArrowParams::_arrowParamsTag, ArrowParams::CreateDefaultInstance, true);
           VizWinMgr::InstallTab(ArrowParams::_arrowParamsTag, ArrowEventRouter::CreateTab);

           // Models
#ifdef MODELS
           ParamsBase::RegisterParamsBaseClass(Transform3d::xmlTag(), Transform3d::CreateDefaultInstance, false);
           ParamsBase::RegisterParamsBaseClass(ModelParams::_modelParamsTag, ModelParams::CreateDefaultInstance, true);
           VizWinMgr::InstallTab(ModelParams::_modelParamsTag, ModelEventRouter::CreateTab);
#endif // MODELS
	}
	//For each class that has a manipulator associated with it, insert the method
	//VizWinMgr::RegisterMouseMode(tag, modeType, manip name, xpm (pixmap) )
	static void InstallExtensionMouseModes(){
		VizWinMgr::RegisterMouseMode(ArrowParams::_arrowParamsTag,1, "Barb rake", arrowrake );
	}
};

