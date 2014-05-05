//	File:		flowlinedata.h
//
//	Author:		Alan Norton
//
//	Date:		December 2006
//
//	Description:	Container classes to help pass flow lines to app
//					
//

#ifndef	FLOWLINEDATA_H
#define	FLOWLINEDATA_H

#include "assert.h"
#include <vapor/common.h>

#define END_FLOW_FLAG 1.e30f
#define STATIONARY_STREAM_FLAG -1.e30f

namespace VAPoR
{
	class VECTOR3;
	class CVectorField;
	class Grid;
	//Base class used for steady flow:
	class FLOW_API FlowLineData {
	public:
		FlowLineData(int numLines, int maxPoints, bool useSpeeds, int direction, bool doRGBAs);	
		virtual ~FlowLineData();
			
		float* getFlowPoint(int lineNum, int pointIndex){
			return flowLineLists[lineNum]+3*pointIndex;
		}
		float* getFlowRGBAs(int lineNum, int pointIndex){
			return ((flowRGBAs == 0) ? 0 : flowRGBAs[lineNum]+4*pointIndex);
		}
		void enableRGBAs() {
			if (!flowRGBAs) {
				flowRGBAs = new float*[nmLines];
				for (int i = 0; i< nmLines; i++){
					flowRGBAs[i] = new float[4*mxPoints];
				}
			}
		}
		float getSpeed(int lineNum, int index){
			assert(speedLists && speedLists[lineNum]);
			return *(speedLists[lineNum] + index);
		}
		float* getFlowStart(int lineNum){
			return getFlowPoint(lineNum, integrationStartPosn);
		}
		int getFlowLength(int lineNum){
			return lineLengths[lineNum];
		}
		//For setting in values, base on integration order, not pointIndex.
		//IntegPos starts at 0, and is pos or negative depending on integration direction
		
		void setFlowPoint(int lineNum, int integPos, float x, float y, float z);
			
		void setSpeed(int lineNum, int integPos, float s){
			assert(speedLists);
			*(speedLists[lineNum]+ integPos+ integrationStartPosn) = s;
		}
		//RGBAs are set according to rendering order, not integration order
		void setRGBA(int lineNum, int pointIndex, float r, float g, float b, float a){
			assert(pointIndex >= 0 && pointIndex < mxPoints);
			assert(lineNum >= 0 && lineNum < nmLines);
			*(flowRGBAs[lineNum]+4*pointIndex) = r;
			*(flowRGBAs[lineNum]+4*pointIndex+1) = g;
			*(flowRGBAs[lineNum]+4*pointIndex+2) = b;
			*(flowRGBAs[lineNum]+4*pointIndex+3) = a;
		}
		void setRGB(int lineNum, int pointIndex, float r, float g, float b){
			assert(pointIndex >= 0 && pointIndex < mxPoints);
			assert(lineNum >= 0 && lineNum < nmLines);
			*(flowRGBAs[lineNum]+4*pointIndex) = r;
			*(flowRGBAs[lineNum]+4*pointIndex+1) = g;
			*(flowRGBAs[lineNum]+4*pointIndex+2) = b;
		}
		void setAlpha(int lineNum, int pointIndex, float alpha){
			assert(pointIndex >= 0 && pointIndex < mxPoints);
			assert(lineNum >= 0 && lineNum < nmLines);
			*(flowRGBAs[lineNum]+4*pointIndex+3) = alpha;
		}

		void releaseSpeeds() {
			if (speedLists) {
				for (int i = 0; i<nmLines; i++){
					if (speedLists[i]) delete [] speedLists[i];
				}
				delete [] speedLists;
				speedLists = 0;
			}
		}
		//This is overridden in pathlinedata:
		virtual int getSeedIndex(int lineNum) { return lineNum;}

		// Establish the first point (for rendering) of the flow line.
		// integPos is + or - depending on whether we are integrating forward or backward
		void setFlowStart(int lineNum, int integPos){
			assert(integPos <= 0);
			int prevStart = startIndices[lineNum];
			startIndices[lineNum] = integrationStartPosn + integPos;
			if (prevStart < 0) lineLengths[lineNum] = 1;
			else {
				lineLengths[lineNum] += (prevStart - integPos - integrationStartPosn);
			}

		}
		void setFlowEnd(int lineNum, int integPos);
		int getSeedPosition(){ return integrationStartPosn;}
			
		bool doSpeeds() {return speedLists != 0;}
		virtual int getNumLines() {return nmLines;}
		int getMaxPoints() {return mxPoints;}
		//maximum length available in either forward or direction:  
		int getMaxLength(int dir); //How far we can integrate in a direction
			
		int getStartIndex(int lineNum) {return startIndices[lineNum];}
		int getEndIndex(int lineNum) {return startIndices[lineNum]+lineLengths[lineNum] -1;}
		
		int getFlowDirection() {return flowDirection;}
		//Following is only to be used before inserting points.  
		void setFlowDirection(int dir) {
			flowDirection = dir;
			if(dir < 0 ) integrationStartPosn = mxPoints -1;
			else if (dir > 0) integrationStartPosn = 0;
			else integrationStartPosn = mxPoints/2;
		}
		//Sample indices along specified flowline.
		//Caller supplies int indexList[numSamples], which is populated by method, returning
		//The number of actual samples <= desiredNumSamples.
		int resampleFieldLines(int* indexList, int desiredNumSamples, int lineNum);

		//Scale all the field lines by a (uniform) factor-triple.
		//Needed if the data has been stretched uniformly
		void scaleLines(const float scalefactor[3]);

		//Realign all the flow lines so that the first point is not END_FLOW_FLAG, if possible
		//Requires direction = 1
		void realignFlowLines();
		//Exit codes are 0, 1, 2, or 3, indicating if/how the line exits the region:
		// 0 = no exit
		// 1 = exit at start
		// 2 = exit at end
		// 3 = exit at start and end
		void setExitCode(int lineNum, int val) {exitCodes[lineNum] = val;}
		void addExit(int lineNum, int code){ exitCodes[lineNum] |= code;}
		bool exitAtStart(int lineNum) {return ((exitCodes[lineNum] & 1)!= 0);}
		bool exitAtEnd(int lineNum) {return ((exitCodes[lineNum] & 2)!= 0);}
		int getExitCode(int lineNum) {return exitCodes[lineNum];}
		
	protected:
		int integrationStartPosn; //Where in the list does the integration begin? 
		int mxPoints;
		int nmLines;
		int flowDirection; //-1, 0, or 1.  Used only on steady flow.
		int* startIndices;
			//startIndices Indicates start position of the 'first' point
			//in the flow line (not the first point for integration).
			//Integration starts at index 0 for forward integration, starts at index
			// mxPoints/2 for bidirectional integration, and starts at last index,
			// startIndices[nline]+mxPoints - 1 for backwards integration
		int* lineLengths;
		int* exitCodes;  //0 = no exit; 1= exit start, 2= exit end, 3= exit start&end
		float** flowLineLists;
		float** speedLists;
		float** flowRGBAs;
	};
	//extension of above, for time-varying flowlines
class FLOW_API PathLineData : public FlowLineData {
	public:
		PathLineData(int numLines, int maxPoints, bool useSpeeds, bool doRGBAs, int minTime, int maxTime, float sampleRate) :
			FlowLineData(numLines, maxPoints, useSpeeds, 0, doRGBAs) {
				//seedTimes = new int[numLines];
				seedIndices = new int[numLines];
				startTimeStep = minTime;
				endTimeStep = maxTime;
				samplesPerTStep = sampleRate;
				actualNumLines = 0;
		}
		~PathLineData() { delete [] seedIndices;}//delete seedTimes?
		void setPointAtTime(int lineNum, float timeStep, float x, float y, float z);
		void setSpeedAtTime(int lineNum, float timeStep, float speed);
		void setFlowStartAtTime(int lineNum, float startTime);
		void setFlowEndAtTime(int lineNum, float endTime);
		void insertSeedAtTime(int seedIndex, int timeStep, float x, float y, float z);
		float* getPointAtTime(int lineNum, float timeStep);

		
		
		//Convert first and last indices to timesteps
		int getFirstTimestep(int lineNum){
			int indx = getStartIndex(lineNum);
			//If it doesn't divide evenly, need to increase...
			return ((int)(0.999f+(float)indx/samplesPerTStep) + startTimeStep);
		}
		int getLastTimestep(int lineNum){
			int indx = getEndIndex(lineNum);
			return ((int)((float)indx/samplesPerTStep + startTimeStep));
		}
		//Get time associated with point in flow, 
		//Doesn't check whether this is a valid point in the flow.
		float getTimeInPath(int lineNum, int index){
			return (index/samplesPerTStep + startTimeStep);
		}
		// Convert timesteps to indices:
		int getIndexFromTimestep(int ts){
			return ((int)(((float)(ts - startTimeStep)*samplesPerTStep) + 0.5f));
		}
		//Override the parent getNumLines() method, because the number of lines is
		//equal to the number of valid seeds inserted, and it
		//may actually be less than nmLines
		virtual int getNumLines() {return actualNumLines;}
		//Determine how many lines exist at a given time step
		int getNumLinesAtTime(int timeStep);
		
		float getSamplesPerTimestep(){return samplesPerTStep;}
		void setSeedIndex(int linenum, int seedIndex){seedIndices[linenum] = seedIndex;}
		int getSeedIndex(int linenum){return seedIndices[linenum];}
		//int getSeedTime(int linenum) {return seedTimes[linenum];}
		//Paths can change their direction...
		void setPathDirection(int dir) { flowDirection = dir;}
	protected:
		int startTimeStep, endTimeStep;
		//int *seedTimes;
		int *seedIndices;//This is the index in the current seedList, for color mapping, etc
		float samplesPerTStep;
		int actualNumLines;
	};
};
#endif

