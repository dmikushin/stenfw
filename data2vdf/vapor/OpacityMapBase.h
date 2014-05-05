//--OpacityMapBase.h ---------------------------------------------------------
//
// Copyright (C) 2006 Kenny Gruchalla.  All rights reserved.
//
// Various types of mappings from opacity to data value. 
// 
//----------------------------------------------------------------------------

#ifndef OpacityMapBase_H
#define OpacityMapBase_H


#include <iostream>
#include <vapor/ExpatParseMgr.h>
#include <vapor/tfinterpolator.h>

namespace VAPoR {

class XmlNode;
class ParamNode;
class MapperFunctionBase;

class PARAMS_API OpacityMapBase : public ParsedXml 
{
  class ControlPoint
  {

  public:

    ControlPoint();
    ControlPoint(float value, float opacity);
    ControlPoint(const ControlPoint &cp);

    void  opacity(float op) { _opacity = op; }   
    float opacity()         { return _opacity; } 

    void  value(float val) { _value = val; }  // Normalized Coordinates
    float value()          { return _value; } // Normalized Coordinates

    void                 type(TFInterpolator::type t) { _type = t; }
    TFInterpolator::type type()                       { return _type; }

    void  select()   { _selected = true; }
    void  deselect() { _selected = false; }
    bool  selected() { return _selected; }
	

  private:

    TFInterpolator::type _type;

    float _value;   // Normalized coordinates
    float _opacity;
    
    bool  _selected;
  };

public:

  enum Type
  {
    CONTROL_POINT,
    GAUSSIAN,
    INVERTED_GAUSSIAN,
    SINE
  };

#ifdef	DEAD
  OpacityMapBase(RenderParams *params, OpacityMapBase::Type type=CONTROL_POINT);
#else
  OpacityMapBase(OpacityMapBase::Type type=CONTROL_POINT);
#endif
  OpacityMapBase(const OpacityMapBase &omap);

  virtual ~OpacityMapBase();

  void clear();

  ParamNode* buildNode();

  const OpacityMapBase& operator=(const OpacityMapBase &cmap);

  float opacity(float value);
  bool  bounds(float value);

  void             type(OpacityMapBase::Type type) { _type = type; }
  OpacityMapBase::Type type() const              { return _type; }

  bool isEnabled() { return _enabled; }
  void setEnabled(bool flag) { _enabled = flag;}

  virtual float minValue() const;      // Data Coordinates
  virtual void  minValue(float value); // Data Coordinates

  virtual float maxValue() const;      // Data Coordinates
  virtual void  maxValue(float value); // Data Coordinates

  int numControlPoints()      { return (int)_controlPoints.size(); }

  void  addNormControlPoint(float normv, float opacity); // Normalized Coords
  void  addControlPoint(float value, float opacity);     // Data Coordinates
  void  deleteControlPoint(int index);
  void  moveControlPoint(int index, float dx, float dy); // Data Coordinates

  float controlPointOpacity(int index);
  void  controlPointOpacity(int index, float opacity);

  float controlPointValue(int index);               // Data Coordinates
  void  controlPointValue(int index, float value);  // Data Coordinates

  void   mean(double mean);                      // Normalized 
  double mean() const          { return _mean; } // Normalized 

  void   sigmaSq(double ssq);                             // Normalized
  double sigmaSq() const       { return normSSq(_ssq); }   // Normalized

  void   sineFreq(double freq);                                  // Normalized
  double sineFreq() const      { return normSineFreq(_freq); }   // Normalized

  void   sinePhase(double p);                                  // Normalized
  double sinePhase() const   { return normSinePhase(_phase); } // Normalized

  void setOpaque();
  bool isOpaque();

  static string xmlTag() { return _tag; }

  virtual bool elementStartHandler(ExpatParseMgr*, int, std::string&, 
                                   const char **attribs);
  virtual bool elementEndHandler(ExpatParseMgr*, int depth, std::string &tag);

protected:

  int leftControlIndex(float val);
  static bool sortCriterion(ControlPoint *p1, ControlPoint *p2);

  double normSSq(double ssq) const;
  double denormSSq(double ssq) const;

  double normSineFreq(double freq) const;
  double denormSineFreq(double freq) const;

  double normSinePhase(double phase) const;
  double denormSinePhase(double phase) const;

  float _minValue;
  float _maxValue;
   
private:

  OpacityMapBase::Type _type;


  bool  _enabled;

  vector<ControlPoint*> _controlPoints;

  double _mean;
  double _ssq;

  double _freq;
  double _phase;

  const double _minSSq;
  const double _maxSSq;
  const double _minFreq;
  const double _maxFreq;
  const double _minPhase;
  const double _maxPhase;

  static const string _tag;
  static const string _minTag;
  static const string _maxTag;
  static const string _enabledTag;
  static const string _meanTag;
  static const string _ssqTag;
  static const string _freqTag;
  static const string _phaseTag;
  static const string _typeTag;
  static const string _controlPointTag;
  static const string _cpOpacityTag;
  static const string _cpValueTag;
};
};

#endif // OpacityMapBase_H
