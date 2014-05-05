//	File:		VaporFlow.h
//
//	Author:		Liya Li
//
//	Date:		July 2005
//
//	Description:	Definition of VaporFlow class. It contains the interface
//					between gui and underlying flow functions.
//

#ifndef	_VaporFlow_h_
#define	_VaporFlow_h_

#include <vapor/DataMgr.h>
#include <vapor/MyBase.h>
#include <vapor/common.h>
#include <vapor/flowlinedata.h>




namespace VAPoR
{
	class VECTOR3;
	class CVectorField;
	class Grid;
	class FlowLineData;
	class PathLineData;
	class FieldData;
	
	class FLOW_API VaporFlow : public VetsUtil::MyBase
	{
	public:
		// constructor and destructor
		VaporFlow(DataMgr* dm = NULL);
		~VaporFlow();
		void Reset(void);

		void SetSteadyFieldComponents(const char* xvar, const char* yvar, const char* zvar);
		void SetUnsteadyFieldComponents(const char* xvar, const char* yvar, const char* zvar);

		void SetRegion(size_t num_xforms, int clevel, const size_t min[3], const size_t max[3], const size_t min_bdim[3], const size_t max_bdim[3], size_t fullGridHeight);
		void SetRakeRegion(const size_t min[3], const size_t max[3], const size_t min_bdim[3], const size_t max_bdim[3]);
		void SetUnsteadyTimeSteps(int timeStepList[], size_t numSteps);
		void SetSteadyTimeSteps(size_t timeStep, int direction){
			steadyStartTimeStep = timeStep;
			steadyFlowDirection = direction;
		}
		
		void ScaleSteadyTimeStepSizes(double userTimeStepMultiplier, double animationTimeStepMultiplier);
		void ScaleUnsteadyTimeStepSizes(double userTimeStepMultiplier, double animationTimeStepMultiplier);
		
		void SetRegularSeedPoints(const double min[3], const double max[3], const size_t numSeeds[3]);
		void SetIntegrationParams(float initStepSize, float maxStepSize);
		void SetDistributedSeedPoints(const double min[3], const double max[3], int numSeeds, 
			const char* xvar, const char* yvar, const char* zvar, float bias);
		
		//New version for API.  Uses rake, then puts integration results in container
		bool GenStreamLines(FlowLineData* container, unsigned int randomSeed);
		//Version for field line advection, takes seeds from unsteady container, 
		//(optionally) prioritizes the seeds
		bool GenStreamLines (FlowLineData* steadyContainer, PathLineData* unsteadyContainer, int timeStep, bool prioritize);
	    
		//Obtains a list of seeds for the currently established rake.
		//Uses settings established by SetRandomSeedPoints, SetDistributedSeedPoints,
		//or SetRegularSeedPoints

		int GenRakeSeeds(float* seeds, int timeStep, unsigned int randomSeed, int stride = 3);

		//Version that actually does the work
		bool GenStreamLinesNoRake(FlowLineData* container, float* seeds);
		
		//Incrementally do path lines:
		bool ExtendPathLines(PathLineData* container, int startTimeStep, int endTimeStep,
			bool doingFLA);

		//Like the above, but put results into fieldLineData array.  
		bool AdvectFieldLines(FlowLineData** containerArray, int startTimeStep, int endTimeStep, int maxNumSamples);

		void SetPeriodicDimensions(bool xPeriodic, bool yPeriodic, bool zPeriodic);
		
		float* GetData(size_t ts, const char* varName);
		
		bool regionPeriodicDim(int i) {return (periodicDim[i] && fullInDim[i]);}
		void SetPriorityField(const char* varx, const char* vary, const char* varz,
			float minField = 0.f, float maxField = 1.e30f);
		//Go through the steady field lines, identify the point on each line with the highest
		//priority.  Insert resulting points into the pathContainer.
		//A unique point is inserted for each nonempty field line (provided it is 
		//inside the current region).
		//Does different things based on whether the container or pathContainer are non-null
		bool prioritizeSeeds(FlowLineData* container, PathLineData* pathContainer, int timestep);
		
		//Methods to encapsulate getting data out of a field.  
		//Returns false if unsuccessful at setting up variables
		//Note that the field is NOT scaled by the current scale factor
		//unless the boolean argument is true
		FieldData* setupFieldData(const char* varx, const char* vary, const char* varz, bool useRakeBounds, int numRefinements, int timestep, bool scaleField);
	
		void releaseFieldData(FieldData*);
		//Obtain min/max vector magnitude in specified region.  Return false on error 
		bool getFieldMagBounds(float* minVal, float* maxVal, const char* varx, const char* vary, const char* varz, 
			bool useRakeBounds, int numRefinements, int timestep);

		DataMgr* getDataMgr(){return dataMgr;}
		size_t getFullGridHeight() {return full_height;}
	 
	private:
		bool Get3Data(size_t ts, const char* xVarName, const char* yVarName, 
			const char* zVarName, float** uData, float ** vData, float **wData);

		size_t userTimeUnit;						// time unit in the original data
		size_t userTimeStep;						// enumerate time steps in source data
		double userTimeStepSize;					// number of userTimeUnits between consecutive steps, which
													// may not be constant
		double animationTimeStepSize;				// successive positions in userTimeUnits
		double steadyUserTimeStepMultiplier;
		double steadyAnimationTimeStepMultiplier;
		double unsteadyUserTimeStepMultiplier;
		double unsteadyAnimationTimeStepMultiplier;
		double animationTimeStep;					// which frame in animation
		double integrationTimeStepSize;				// used for integration

		size_t steadyStartTimeStep;					// refer to userTimeUnit.  Used only for
		size_t endTimeStep;							// steady flow
		size_t timeStepIncrement;
		int steadyFlowDirection;					// -1, 0 or 1

		int* unsteadyTimestepList;
		size_t numUnsteadyTimesteps;

		float minRakeExt[3];						// minimal rake range 
		float maxRakeExt[3];						// maximal rake range
		size_t minBlkRake[3], maxBlkRake[3];
		size_t minRake[3], maxRake[3];
		size_t numSeeds[3];							// number of seeds
		bool periodicDim[3];						// specify the periodic dimensions
		bool fullInDim[3];							// determine if the current region is full in each dimension
		bool bUseRandomSeeds;						// whether use randomly or regularly generated seeds

		float initialStepSize;						// for integration
		float maxStepSize;

		DataMgr* dataMgr;							// data manager
		char *xSteadyVarName, *ySteadyVarName, *zSteadyVarName;		
													// name of three variables for steady field
		char *xUnsteadyVarName, *yUnsteadyVarName, *zUnsteadyVarName;		
													// name of three variables for unsteady field
		char *xPriorityVarName, *yPriorityVarName, *zPriorityVarName;
													// field variables used for prioritizing seeds on flowlines
		char *xSeedDistVarName, *ySeedDistVarName, *zSeedDistVarName;
													// field variables used to determine random seed distribution
		size_t numXForms, minBlkRegion[3], maxBlkRegion[3];// in block coordinate
		int compressLevel;
		size_t minRegion[3], maxRegion[3];			//Actual region bounds
		float flowPeriod[3];						//Used if data is periodic
		float* flowLineAdvectionSeeds;
		float minPriorityVal, maxPriorityVal;
		float seedDistBias;
		size_t full_height;  //0 unless grid is layered.
	};
};

#endif

