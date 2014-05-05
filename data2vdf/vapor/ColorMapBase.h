//--ColorMapBase.h ---------------------------------------------------------
//
// Copyright (C) 2006 Kenny Gruchalla.  All rights reserved.
//
// A map from data value to/from color.
// 
//----------------------------------------------------------------------------

#ifndef ColorMapBase_H
#define ColorMapBase_H

#include <vapor/ExpatParseMgr.h>
#include <vapor/tfinterpolator.h>

namespace VAPoR {

class ParamNode;

class PARAMS_API ColorMapBase : public ParsedXml 
{

public:

  class PARAMS_API Color
  {

  public:
   
    Color();
    Color(float h, float s, float v);
    Color(const Color &color);

    void toRGB(float *rgb);

    void  hue(float h) { _hue = h; }
    float hue()        { return _hue; }

    void  sat(float s) { _sat = s; }
    float sat()        { return _sat; }

    void  val(float v) { _val = v; }
    float val()        { return _val; }

  private:

    float _hue;
    float _sat;
    float _val;
  };


  ColorMapBase();
  ColorMapBase(const ColorMapBase &cmap);

  virtual ~ColorMapBase();

  const ColorMapBase& operator=(const ColorMapBase &cmap);

  ParamNode* buildNode();

  void  clear();

  virtual float minValue() const;      // Data Coordinates
  virtual void  minValue(float value); // Data Coordinates

  virtual float maxValue() const;      // Data Coordinates
  virtual void  maxValue(float value); // Data Coordinates

  int numControlPoints()                { return (int)_controlPoints.size(); }

  Color controlPointColor(int index);
  void  controlPointColor(int index, Color color);

  float controlPointValue(int index);               // Data Coordinates
  void  controlPointValue(int index, float value);  // Data Coordinates

  void addControlPointAt(float value);
  void addControlPointAt(float value, Color color);
  void addNormControlPoint(float normValue, Color color);
  void deleteControlPoint(int index);

  void move(int index, float delta);
  
  Color color(float value);

  static string xmlTag() { return _tag; }

  virtual bool elementStartHandler(ExpatParseMgr*, int, std::string&, 
                                   const char **attribs);
  virtual bool elementEndHandler(ExpatParseMgr*, int depth, std::string &tag);
 
	
protected:

  int leftIndex(float val);

  class ControlPoint
  {

  public:

    ControlPoint();
    ControlPoint(Color c, float v);
    ControlPoint(const ControlPoint &cp);

    void  color(Color color) { _color = color; }
    Color color()            { return _color; }

    void  value(float val) { _value = val; }
    float value()          { return _value; }

    void                 type(TFInterpolator::type t) { _type = t; }
    TFInterpolator::type type()                       { return _type; }

    void  select()   { _selected = true; }
    void  deselect() { _selected = false; }
    bool  selected() { return _selected; }

  private:

    TFInterpolator::type _type;

    float _value;
    Color _color;
    
    bool  _selected;
  };

  static bool sortCriterion(ControlPoint *p1, ControlPoint *p2);

  float _minValue;
  float _maxValue;


private:


  vector<ControlPoint*> _controlPoints;

  static const string _tag;
  static const string _minTag;
  static const string _maxTag;
  static const string _controlPointTag;  
  static const string _cpHSVTag;
  static const string _cpRGBTag;
  static const string _cpValueTag;
};
};

#endif // ColorMapBase_H
