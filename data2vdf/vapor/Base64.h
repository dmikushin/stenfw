
#ifndef	_Base64_h_
#define	_Base64_h_
#include <vapor/MyBase.h>

namespace VetsUtil {

class COMMON_API Base64 : public MyBase {

public:


 Base64();
 ~Base64();
 void Encode(const unsigned char *input, size_t n, string &output);
 void EncodeStreamBegin(string &output);
 void EncodeStreamNext(const unsigned char *input, size_t n, string &output);
 void EncodeStreamEnd(string &output);

 //! Size returned is guaranteed to be large enough (maybe larger)
 //!
 size_t GetEncodeSize(size_t n);
 int Decode(const string &input, unsigned char *output, size_t *n);

private:
	unsigned char _eTable[64];	// encoding table
	unsigned char _dTable[256];	// decoding table
	int _maxLineLength;
	int _ngrps; // num encoded output characters for current line
	unsigned char _inbuf[3];	// input buffer
	int _inbufctr;	// num bytes in input buffer

};

};

#endif
