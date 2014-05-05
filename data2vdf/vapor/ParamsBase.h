//************************************************************************
//									*
//		     Copyright (C)  2008				*
//     University Corporation for Atmospheric Research			*
//		     All Rights Reserved				*
//									*
//************************************************************************/
//
//	File:		ParamsBase.h
//
//	Author:		John Clyne, modified by Alan Norton
//			National Center for Atmospheric Research
//			PO 3000, Boulder, Colorado
//
//	Date:		March 2008
//
//	Description:	
//		Defines the ParamsBase class
//		This is an abstract class for classes that rely on 
//		accessing an XML node for get/set
//

#ifndef ParamsBase_H
#define ParamsBase_H

#include <vapor/XmlNode.h>
#include <vapor/ParamNode.h>
#include <vapor/ExpatParseMgr.h>
#include "assert.h"
#include <vapor/common.h>


using namespace VetsUtil;

namespace VAPoR{

//
//! \class ParamsBase
//! \brief Nodes with state in Xml tree representation
//! \author John Clyne
//! \version $Revision: 1.17.2.1 $
//! \date    $Date: 2011/12/19 20:27:04 $
//!
//! This is abstract parent of Params and related classes with state 
//! kept in an xml node.  Used with the ParamNode class to support
//! user-defined Params classes as well as other classes such as
//! the TransferFunction class.
//! 
//! Users can extend ParamsBase classes to include arbitrary
//! child (sub) nodes, by support parsing of such nodes. 
//!

class XmlNode;
class ParamNode;
class ParamsBase;

class PARAMS_API ParamsBase : public ParsedXml {
	

typedef ParamsBase* (*BaseCreateFcn)();

public: 
ParamsBase(
	XmlNode *parent, const string &name
 );
typedef int ParamsBaseType;
//! Default constructor 
ParamsBase(const string& name) {
	_currentParamNode = _rootParamNode = 0;
	_parseDepth = 0;
	_paramsBaseName = name;
}
//! Copy constructor.  
ParamsBase(const ParamsBase &pbase);

virtual ~ParamsBase();

 //! Make a copy of a ParamBase that optionally uses specified 
 //! clone of the ParamNode as its root node.  If the root
 //! is null, the copy ignores any ParamNodes.  The default implementation
 //! is sufficient for ParamsBase classes that are built from
 //! a ParamNode hierarchy.
 //!
 //! \param[in] newRoot Root of cloned ParamsBase instance
 //! \retval instance Pointer to cloned instance
 //
 virtual ParamsBase* deepCopy(ParamNode* newRoot = 0);
 
 //! Set the parent node of the XmlNode tree.
 //!
 //! Sets a new parent node for the XmlNode tree parameter 
 //! data base. The parent node, \p parent, must have been
 //! previously initialized by passing it as an argument to
 //! the class constructor ParamBase(). This method permits
 //! wholesale changing of the entire parameter database.
 //!
 //! \param[in] parent Parent XmlNode.
 //
 void SetParent(XmlNode *parent);


 //! Xml start tag parsing method
 //!
 //! This method is called to handle parsing of an XML file. The contents
 //! of the file will replace any current parameter settings. The method
 //! is virtual so that derived classes may receive notification when
 //! an object instance is reseting state from an XML file
 //!
 //! Override this method if you are not using the ParamNode API to
 //! specify the state in terms of
 //! the xml representation of the class
 //! \sa elementEndHandler
 //! \retval status False indicates parse error
 //
 virtual bool elementStartHandler(
	ExpatParseMgr* pm, int depth, string& tag, const char ** attribs
 );

 //! Xml end tag parsing method
 //!
 //! This method is called to handle parsing of an XML file. The contents
 //! of the file will replace any current parameter settings. The method
 //! is virtual so that derived classes may receive notification when
 //! an object instance has finished reseting state from an XML file.
 //! Override the default method if the class is not based on
 //! a hierarchy of ParamNode objects representing 
 //! the XML representation of the class
 //! \sa elementStartHandler
 //! \retval status False indicates parse error
 //
 virtual bool elementEndHandler(ExpatParseMgr* pm, int depth, string& tag);

 //! Return the top (root) of the parameter node tree
 //!
 //! This method returns the top node in the parameter node tree
 //!

ParamNode *GetRootNode() { return(_rootParamNode); }

//!	
//! Method to build an xml node from state.
//! This only needs to be implemented if the state of the ParamsBase
//! is not specified by the root ParamNode 
//! \retval node ParamNode representing the current ParamsBase instance
//!

virtual ParamNode* buildNode(); 

//!	
//! Method for manual setting of node flags
//!

void SetFlagDirty(const string& flag);
//!	
//! Method for obtaining the name and/or tag associated with the instance
//!

const string& GetName() {return _paramsBaseName;}
//!	
//! Method for obtaining the type Id associated with a ParamsBase instance
//! \retval int ParamsBase TypeID for ParamsBase instance 
//!

ParamsBaseType GetParamsBaseTypeId() {return GetTypeFromTag(_paramsBaseName);}

//!
//! Static method for converting a Tag to a ParamsBase typeID
//! \retval int ParamsBase TypeID for Tag
//!
static ParamsBaseType GetTypeFromTag(const string&tag);

//!
//! Static method for converting a ParamsBase typeID to a Tag
//! \retval string Tag (Name) associated with ParamsBase TypeID
//!
static const string& GetTagFromType(ParamsBaseType t);

//!
//! Static method for constructing a default instance of a ParamsBase 
//! class based on the typeId.
//! \param[in] pType TypeId of the ParamsBase instance to be created.
//! \retval instance newly created ParamsBase instance
//!
static ParamsBase* CreateDefaultParamsBase(int pType){
	ParamsBase *p = (createDefaultFcnMap[pType])();
	return p;
}
//!
//! Static method for constructing a default instance of a ParamsBase 
//! class based on the Tag.
//! \param[in] tag XML tag of the ParamsBase instance to be created.
//! \retval instance newly created ParamsBase instance
//!
static ParamsBase* CreateDefaultParamsBase(const string&tag);

//Methods for registration and tabulation of existing Params instances


//!
//! Static method for registering a ParamsBase class.
//! This calls CreateDefaultInstance on the class. 
//! \param[in] tag  Tag of class to be registered
//! \param[in] fcn  Method that creates default instance of ParamsBase class 
//! \param[in] isParams set true if the ParamsBase class is derived from Params
//! \retval classID Returns the ParamsBaseClassId, or 0 on failure 
//!
	static int RegisterParamsBaseClass(const string& tag, BaseCreateFcn fcn, bool isParams);
//!
//! Static method for registering a tag for an already registered ParamsBaseClass.
//! This is needed for backwards compatibility when a tag is changed.
//! The class must first be registered with the new tag.
//! \param[in] tag  Tag of class to be registered
//! \param[in] newtag  Previously registered tag (new name of class)
//! \param[in] isParams set true if the ParamsBase class is derived from Params
//! \retval classID Returns the ParamsBaseClassId, or 0 on failure 
//!
	static int ReregisterParamsBaseClass(const string& tag, const string& newtag, bool isParams);
//!
//! Specify the Root ParamNode of a ParamsBase instance 
//! \param[in] pn  ParamNode of new root 
//!
	virtual void SetRootParamNode(ParamNode* pn){_rootParamNode = pn;}
//!
//! Static method to determine how many Params classes are registered
//! \retval count Number of registered Params classes
//!
	static int GetNumParamsClasses() {return numParamsClasses;}
//!
//! Static method to determine if a ParamsBase class is a Params class
//! \param[in] tag XML tag associated with ParamsBase class
//! \retval status True if the specified class is a Params class
//!
	static bool IsParamsTag(const string&tag) {return (GetTypeFromTag(tag) > 0);}
#ifndef DOXYGEN_SKIP_THIS
	static ParamsBase* CreateDummyParamsBase(std::string tag);
	
	static void addDummyParamsBaseInstance(ParamsBase*const & pb ) {dummyParamsBaseInstances.push_back(pb);}

	static void clearDummyParamsBaseInstances();
#endif
private:
	//These should be accessed by subclasses through get() and set() methods
	ParamNode *_currentParamNode;
	ParamNode *_rootParamNode;
	

protected:
	static vector<ParamsBase*> dummyParamsBaseInstances;
	static const string _emptyString;
	virtual ParamNode *getCurrentParamNode() {return _currentParamNode;}
	
	virtual void setCurrentParamNode(ParamNode* pn){ _currentParamNode=pn;}
	
	
	
	static map<string,int> classIdFromTagMap;
	static map<int,string> tagFromClassIdMap;
	static map<int,BaseCreateFcn> createDefaultFcnMap;

	string _paramsBaseName;
	int _parseDepth;
	static int numParamsClasses;
	static int numEmbedClasses;


protected:

 //! Return the current node in the parameter node tree
 //!
 //! This method returns the current node in the parameter node tree. 
 //! \sa Push(), Pop()
 //! \retval node Current ParamNode
 //!
 ParamNode *GetCurrentNode() { return(_currentParamNode); }

 //! Move down a level in the parameter tree
 //!
 //! The underlying storage model for parameter data is a hierarchical tree. 
 //! By default the hierarchy is flat. This method can be used to add
 //! and navigate branches of the tree. Invoking this method makes the branch
 //! named by \p name the current branch (node). If the branch \p name does not
 //! exist it will be created with the name provided. 
 //! Subsequent set and get methods will operate
 //! relative to the current branch.
 //! User-specific subtrees can be provided by extending this method
 //!
 //! \param[in] tag The name of the branch
 //! \param[in] pBase optional ParamsBase object to be associated with ParamNode
 //! \retval node Returns the new current node
 //!
 //! \sa Pop(), Delete(), GetCurrentNode()
 //
 ParamNode *Push(
	 string& tag,
	 ParamsBase* pBase = 0
	);


 //! Move up one level in the paramter tree
 //!
 //! This method move back up the tree hierarchy by one level.
 //! Moving up past the root of the tree is prohibited and will silenty fail
 //! with no ill effects.
 //!
 //! \retval node Returns the new current node
 //!
 //! \sa Pop(), Delete()
 //
 ParamNode *Pop();

 //! Delete the named branch.
 //!
 //! This method deletes the named child, and all decendents, of the current 
 //! destroying it's contents in the process. The 
 //! named node must be a child of the current node. If the named node
 //! does not exist the result is a no-op.
 //!
 //! \param[in] name The name of the branch
 //
 void Remove(const string &name);

 //! Return the attributes associated with the current branch
 //!
 //! \retval map attribute mapping
 //
 const map <string, string> &GetAttributes();


 //! Remove (undefine) all parameters
 //!
 //! This method deletes any and all paramters contained in the base 
 //! class as well as deleting any tree branches.
 //
 void Clear();
};
#ifndef DOXYGEN_SKIP_THIS
//The DummyParamsBase is simply holding the parse information for
//A paramsBase extension class that is not present. This can only occur
//as a ParamsBase node inside a DummyParams node
class DummyParamsBase : public ParamsBase {
	public:
		DummyParamsBase(XmlNode *parent, const string &name) :
		  ParamsBase(parent, name) {}
	virtual ~DummyParamsBase(){}
};
#endif //DOXYGEN_SKIP_THIS
}; //End namespace VAPoR
#endif //ParamsBase_H
