//
//      $Id: VDFIOBase.h,v 1.12 2010/09/13 18:08:05 clynejp Exp $
//


#ifndef	_VDFIOBase_h_
#define	_VDFIOBase_h_

#include <cstdio>
#include <vapor/MyBase.h>
#include <vapor/MetadataVDC.h>

namespace VAPoR {


//
//! \class VDFIOBase
//! \brief Abstract base class for performing data IO to a VDC
//! \author John Clyne
//! \version $Revision: 1.12 $
//! \date    $Date: 2010/09/13 18:08:05 $
//!
//! This class provides an API for performing low-level IO 
//! to/from a Vapor Data Collection (VDC)
//
class VDF_API	VDFIOBase : public MetadataVDC {

public:

 //! Constructor for the VDFIOBase class.
 //! \param[in] metadata Pointer to a metadata class object for which all
 //! future class operations will apply
 //! \note The success or failure of this constructor can be checked
 //! with the GetErrCode() method.
 //!
 //! \sa Metadata, GetErrCode(),
 //
 VDFIOBase(
	const MetadataVDC &metadata
 );

 //! Constructor for the VDFIOBase class.
 //! \param[in] metafile Path to a metadata file for which all
 //! future class operations will apply
 //! \note The success or failure of this constructor can be checked
 //! with the GetErrCode() method.
 //!
 //! \sa Metadata, GetErrCode(),
 //
 VDFIOBase(
	const string &metafile
 );

 virtual ~VDFIOBase();

 //! Return the read timer
 //!
 //! This method returns the accumulated clock time, in seconds, 
 //! spent reading data from files. 
 //!
 double	GetReadTimer() const { return(_read_timer_acc); };

 //! Return the seek timer
 //!
 //! This method returns the accumulated clock time, in seconds, 
 //! spent performing file seeks (in general this is zero)
 //!
 double	GetSeekTimer() const { return(_seek_timer_acc); };
 void SeekTimerReset() {_seek_timer_acc = 0;};
 void SeekTimerStart() {_seek_timer = GetTime();};
 void SeekTimerStop() {_seek_timer_acc += (GetTime() - _seek_timer);};

 //! Return the write timer
 //!
 //! This method returns the accumulated clock time, in seconds, 
 //! spent writing data to files. 
 //!
 double	GetWriteTimer() const { return(_write_timer_acc); };

 //! Return the transform timer
 //!
 //! This method returns the accumulated clock time, in seconds, 
 //! spent transforming data. 
 //!
 double	GetXFormTimer() const { return(_xform_timer_acc); };

 double GetTime() const;

 virtual int    BlockReadRegion(
    const size_t /*bmin*/[3], const size_t /*bmax*/[3],
    float * /*region*/, int /*unblock*/ = 1
 ) { return(-1);};

 virtual const float *GetDataRange() const = 0;

protected:
 void _ReadTimerReset() {_read_timer_acc = 0;};
 void _ReadTimerStart() {_read_timer = GetTime();};
 void _ReadTimerStop() {_read_timer_acc += (GetTime() - _read_timer);};

 void _WriteTimerReset() {_write_timer_acc = 0;};
 void _WriteTimerStart() {_write_timer = GetTime();};
 void _WriteTimerStop() {_write_timer_acc += (GetTime() - _write_timer);};

 void _XFormTimerReset() {_xform_timer_acc = 0;};
 void _XFormTimerStart() {_xform_timer = GetTime();};
 void _XFormTimerStop() {_xform_timer_acc += (GetTime() - _xform_timer);};


private:

 double	_read_timer_acc;
 double	_write_timer_acc;
 double	_seek_timer_acc;
 double	_xform_timer_acc;

 double	_read_timer;
 double	_write_timer;
 double	_seek_timer;
 double	_xform_timer;

 int	_VDFIOBase();

};


 int	MkDirHier(const string &dir);
 void	DirName(const string &path, string &dir);

}

#endif	//	
