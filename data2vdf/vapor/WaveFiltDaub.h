#include <string>
#include <vapor/WaveFiltBase.h>

using namespace std;

#ifndef	_WaveFiltDaub_h_
#define	_WaveFiltDaub_h_

namespace VAPoR {

//
//! \class WaveFiltDaub
//! \brief Daubechies family FIR filters
//! \author John Clyne
//! \version $Revision: 1.1 $
//! \date    $Date: 2010/06/07 16:43:10 $
//!
//! This class provides FIR filters for the Daubechies family of wavelets
//!
class WaveFiltDaub : public WaveFiltBase {

public:

 //! Create a set of Daubechies filters
 //!
 //! \param[in] wavename The Daubechies family wavelet member. Valid values
 //! are "db", "db", "db", "db", "db5", "db6", "db7", "db8", "db9", and 
 //! "db10"
 //!
 WaveFiltDaub(const string &wavename);
 virtual ~WaveFiltDaub();
	

private:
 void _analysis_initialize (int member);
 void _synthesis_initialize (int member);
};

}

#endif
