//************************************************************************
//																		*
//		     Copyright (C)  2004										*
//     University Corporation for Atmospheric Research					*
//		     All Rights Reserved										*
//																		*
//************************************************************************/
//
//	File:		tfinterpolator.h
//
//	Author:		Alan Norton
//			National Center for Atmospheric Research
//			PO 3000, Boulder, Colorado
//
//	Date:		November 2004
//
//	Description:	Defines the TFInterpolator class:   
//		A class to interpolate transfer function values
//		Currently only supports linear interpolation
//
#ifndef TFINTERPOLATOR_H
#define TFINTERPOLATOR_H
#include "math.h"
#include <vapor/common.h>
namespace VAPoR {
class PARAMS_API TFInterpolator{
public:
	//Default is linear
	enum type {
		linear,
		discrete,
		logarithm,
		exponential
	};
	//Determine the interpolated value at intermediate value 0<=r<=1
	//where the value at left and right endpoint is known
	//This method is just a stand-in until we get more sophistication
	//
	static float interpolate(type, float leftVal, float rightVal, float r);
		
	//Linear interpolation for circular (hue) fcn.  values in [0,1).
	//If it's closer to go around 1, then do so
	//
	static float interpCirc(type t, float leftVal, float rightVal, float r);	
};

};

#endif //TFINTERPOLATOR_H

