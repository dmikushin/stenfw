//************************************************************************
//																		*
//		     Copyright (C)  2004										*
//     University Corporation for Atmospheric Research					*
//		     All Rights Reserved										*
//																		*
//************************************************************************/
//					
//	File:		MapperFunctionBase.h
//
//	Author:		Alan Norton
//			National Center for Atmospheric Research
//			PO 3000, Boulder, Colorado
//
//	Date:		August 2005
//
//	Description:	Defines the MapperFunctionBase class  
//		This is the mathematical definition of a function
//		that can be used to map data to either colors or opacities
//		Subclasses can either implement color or transparency 
//		mapping (or both)

#ifndef MAPPERFUNCTIONBASE_H
#define MAPPERFUNCTIONBASE_H
#define MAXCONTROLPOINTS 50
#include <iostream>
#include <vapor/ExpatParseMgr.h>
#include <vapor/OpacityMapBase.h>
#include <vapor/ColorMapBase.h>
#include <vapor/tfinterpolator.h>
#include <vapor/ParamsBase.h>

namespace VAPoR {
class XmlNode;
class ParamNode;
class PARAMS_API MapperFunctionBase : public ParamsBase 
{

public:
	MapperFunctionBase(const string& name);
	MapperFunctionBase(int nBits, const string& name);
	MapperFunctionBase(const MapperFunctionBase &mapper);
	
	virtual ~MapperFunctionBase();

    //
    // Function values
    //
	float opacityValue(float point);
    void  hsvValue(float point, float* h, float* sat, float* val);

	void setOpaque();
	bool isOpaque();

    //
	// Build a lookup table[numEntries][4]
	// (Caller must pass in an empty array)
	//
	void makeLut(float* clut);

    //
    // Data Bounds
    //
	float getMinColorMapValue() { return minColorMapBound; }
	float getMaxColorMapValue() { return maxColorMapBound; }
	float getMinOpacMapValue()  { return minOpacMapBound; }
	float getMaxOpacMapValue()  { return maxOpacMapBound; }

	void setMinColorMapValue(float val) { minColorMapBound = val; }
	void setMaxColorMapValue(float val) { maxColorMapBound = val; }
	void setMinOpacMapValue(float val)  { minOpacMapBound = val; }
	void setMaxOpacMapValue(float val)  { maxOpacMapBound = val; }

    //
    // Variables
    //
   
    int getColorVarNum() { return colorVarNum; }
    int getOpacVarNum() { return opacVarNum; }
   
    virtual void setVarNum(int var)     { colorVarNum = var; opacVarNum = var;}
    virtual void setColorVarNum(int var){ colorVarNum = var; }
    virtual void setOpacVarNum(int var) { opacVarNum = var; }

    //
    // Opacity Maps
    //
    virtual OpacityMapBase* createOpacityMap(
		OpacityMapBase::Type type=OpacityMapBase::CONTROL_POINT
	);
    virtual OpacityMapBase* getOpacityMap(int index) const;
    void        deleteOpacityMap(OpacityMapBase *omap);
    int         getNumOpacityMaps() const { return (int)_opacityMaps.size(); }

    //
    // Opacity scale factor (scales all opacity maps)
    //
	void setOpacityScaleFactor(float val) { opacityScaleFactor = val; }
	float getOpacityScaleFactor()         { return opacityScaleFactor; }

    //
    // Opacity composition
    //
    enum CompositionType
    {
      ADDITION = 0,
      MULTIPLICATION = 1
    };

    void setOpacityComposition(CompositionType t) { _compType = t; }
    CompositionType getOpacityComposition() { return _compType; }

    //
    // Colormap
    //
    virtual ColorMapBase*   getColormap() const;

    //
	// Color conversion 
    //
	static void hsvToRgb(float* hsv, float* rgb);
	static void rgbToHsv(float* rgb, float* hsv);
	
	//
    // Map a point to the specified range, and quantize it.
	//
	static int mapPosition(float x, float minValue, float maxValue, int hSize);

	int getNumEntries()         { return numEntries; }
	void setNumEntries(int val) { numEntries = val; }

    //
	// Map and quantize a real value to the corresponding table index
	// i.e., quantize to current Mapper function domain
	//
	int mapFloatToColorIndex(float point) 
    {
		
      int indx = mapPosition(point, 
                             getMinColorMapValue(), 
                             getMaxColorMapValue(), numEntries-1);
      if (indx < 0) indx = 0;
      if (indx > numEntries-1) indx = numEntries-1;
      return indx;
	}

	float mapColorIndexToFloat(int indx)
    {
      return (float)(getMinColorMapValue() + 
                     ((float)indx)*(float)(getMaxColorMapValue()-
                                           getMinColorMapValue())/
                     (float)(numEntries-1));
	}

	int mapFloatToOpacIndex(float point) 
    {
      int indx = mapPosition(point, 
                             getMinOpacMapValue(), 
                             getMaxOpacMapValue(), numEntries-1);
      if (indx < 0) indx = 0;
      if (indx > numEntries-1) indx = numEntries-1;
      return indx;
	}

	float mapOpacIndexToFloat(int indx)
    {
      return (float)(getMinOpacMapValue() + 
                     ((float)indx)*(float)(getMaxOpacMapValue()-
                                           getMinOpacMapValue())/
                     (float)(numEntries-1));
	}
	
    //
	// Methods to save and restore Mapper functions.
	// The gui opens the FILEs that are then read/written
	// Failure results in false/null pointer
	//
	// These methods are the same as the transfer function methods,
	// except for specifying separate color and opacity bounds,
	// and not having a name attribute
    //
	
	virtual ParamNode* buildNode(); 

	virtual bool elementStartHandler(ExpatParseMgr*, int depth, 
                                     std::string&, const char **);

	virtual bool elementEndHandler(ExpatParseMgr*, int, std::string&);
   
    std::string getName() { return mapperName; }

	// Mapper function tag is public, visible to flowparams
	static const string _mapperFunctionTag;

protected:
    //
	// Set to starting values
	//
	virtual void init();  
    	
		
protected:

    vector<OpacityMapBase*>  _opacityMaps;
    CompositionType      _compType;

    ColorMapBase            *_colormap;

    //
    // XML tags
    //
	static const string _leftColorBoundAttr;
	static const string _rightColorBoundAttr;
	static const string _leftOpacityBoundAttr;
	static const string _rightOpacityBoundAttr;
    static const string _opacityCompositionAttr;
	//Several attributes became tags after version 1.5:
	static const string _leftColorBoundTag;
	static const string _rightColorBoundTag;
	static const string _leftOpacityBoundTag;
	static const string _rightOpacityBoundTag;
    static const string _opacityCompositionTag;
	static const string _hsvAttr;
	static const string _positionAttr;
	static const string _opacityAttr;
	static const string _opacityControlPointTag;
	static const string _colorControlPointTag;
	// Additional attributes not yet supported:
	static const string _interpolatorAttr;
	static const string _rgbAttr;
	
    //
    // Mapping bounds
    //
	float minColorMapBound, maxColorMapBound;
	float minOpacMapBound, maxOpacMapBound;

    //
    // Parent params
    //
#ifdef	DEAD
	RenderParams* myParams;
#endif
	
    //
	// Size of lookup table.  Always 1<<8 currently!
	//
	int numEntries;

    //
	// Mapper function name, if it's named.
    //
	string mapperName;

	//
    // Corresponding var nums
    //
	int colorVarNum;
    int opacVarNum;	

    //
    // Opacity scale factor
    //
    float opacityScaleFactor;
};
};
#endif //MAPPERFUNCTIONBASE_H
