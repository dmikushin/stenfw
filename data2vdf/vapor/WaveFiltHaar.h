
#ifndef	_WaveFiltHaar_h_
#define	_WaveFiltHaar_h_

namespace VAPoR {

//
//! \class WaveFiltHaar
//! \brief Haar FIR filters
//! \author John Clyne
//! \version $Revision: 1.1 $
//! \date    $Date: 2010/06/07 16:43:10 $
//!
//! This class provides FIR filters for the Haar wavelet
//!
class WaveFiltHaar : public WaveFiltBase {

public:
 //! Create a set of Haar wavelet filters
 //!
 WaveFiltHaar();
 virtual ~WaveFiltHaar();
	

private:
 void _analysis_initialize();
 void _synthesis_initialize();
};

}

#endif
