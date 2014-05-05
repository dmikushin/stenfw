//
//


#ifndef	EXPATPARSEMGR_H
#define	EXPATPARSEMGR_H

#include <stack>
#include <expat.h>
#include <vector>
#include <vapor/MyBase.h>
#include <vapor/common.h>
#ifdef WIN32
#pragma warning(disable : 4251)
#endif

namespace VAPoR {
class ExpatParseMgr;
class XmlNode;
//abstract base class for classes that do xml parsing
class VDF_API ParsedXml {
public:
	virtual ~ParsedXml(){}
	//Callbacks from xml parse
	//Start and end handlers return false on failure.
	virtual bool elementStartHandler(ExpatParseMgr*, int /* depth*/ , std::string& /*tag*/, const char ** /*attribs*/) = 0;
	virtual bool elementEndHandler(ExpatParseMgr*, int /*depth*/ , std::string& /*tag*/) = 0;
	virtual bool charHandler (ExpatParseMgr*, const XML_Char *, int ) {return true;}
	
	//Store previous class for stack
	ParsedXml* previousClass;

protected:
	// known xml attribute values
	//
	static const string _stringType;
	static const string _longType;
	static const string _doubleType;
	static const string _typeAttr;

};
// Structure used for parsing metadata files
//
class VDF_API ExpatStackElement {
	public:
	string tag;			// xml element tag
	string data_type;	// Type of element data (string, double, or long)
	int has_data;		// does the element have data?
	int user_defined;	// is the element user defined?
};
class VDF_API ExpatParseMgr : public VetsUtil::MyBase {
public:

	ExpatParseMgr(ParsedXml* topLevelClass);
 
	~ExpatParseMgr();

	void parse(ifstream& is);
	ExpatStackElement* getStateStackTop() {return _expatStateStack.top();}
	// Report an XML parsing error
	//
	void parseError(const char *format, ...);
	vector<long>& getLongData() {return _expatLongData;}
	vector<double>& getDoubleData() {return _expatDoubleData;}
	string& getStringData() {return _expatStringData;}
	//Following two methods are to allow different classes to be created during one
	//XML parsing.  These help maintain a stack of classes.
	//When the parsing is to be passed to another class, the original class must call
	//pushClassStack when the xml for the new class is encountered in the startElementHandler.
	
	void pushClassStack(ParsedXml* pc) {
		pc->previousClass = currentParsedClass;
		currentParsedClass = pc;
	}
	ParsedXml* popClassStack(){
		currentParsedClass = currentParsedClass->previousClass;
		return currentParsedClass;
	}

	void skipElement(string tag, int depth);
	
protected:
	ParsedXml* currentParsedClass;
	
	stack<VDF_API ExpatStackElement *> _expatStateStack;

	XML_Parser _expatParser;	// XML Expat parser handle
	string _expatStringData;	// temp storage for XML element character data
	vector <long> _expatLongData;	// temp storage for XML long data
	vector <double> _expatDoubleData;	// temp storage for XML double  data
	 

	// known xml attribute values
	//
	static const string _stringType;
	static const string _longType;
	static const string _doubleType;

#ifdef	DEAD
	// XML Expat element handlers
	friend void	_StartElementHandler(
		void *userData, const XML_Char *tag, const XML_Char **attrs
	) {
		ExpatParseMgr* mgr = (ExpatParseMgr *) userData;
		mgr->_startElementHandler(tag, attrs);
	}


	friend void _EndElementHandler(void *userData, const XML_Char *tag) {
		ExpatParseMgr* mgr = (ExpatParseMgr *) userData;
		mgr->_endElementHandler(tag);
	}

	friend void	_CharDataHandler(
		void *userData, const XML_Char *s, int len
	) {
		ExpatParseMgr* mgr = (ExpatParseMgr *) userData;
		mgr->_charDataHandler(s, len);
	}

#else
	// XML Expat element handlers
	friend void	_StartElementHandler(
		void *userData, const XML_Char *tag, const XML_Char **attrs
	);


	friend void _EndElementHandler(void *userData, const XML_Char *tag);

	friend void	_CharDataHandler(
		void *userData, const XML_Char *s, int len
	);

#endif
	void _startElementHandler(const XML_Char *tag, const char **attrs);
	void _endElementHandler(const XML_Char *tag);
	void _charDataHandler(const XML_Char *s, int len);

	
	//Function pointers that handle custom parsing
private:
	bool _skipFlag;
	string _skipTag;
	int _skipDepth;

};


};

#endif	//	EXPATPARSEMGR_H
