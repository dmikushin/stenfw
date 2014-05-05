#include <iostream>
#include <fstream>
#include <string.h>
#include <vector>
#include <sstream>

#include <vapor/OptionParser.h>
#include <vapor/MetadataVDC.h>
#include <vapor/MetadataSpherical.h>
#include <vapor/CFuncs.h>
#include <vapor/WaveCodecIO.h>
#ifdef WIN32
#pragma warning(disable : 4996)
#endif
using namespace VetsUtil;
using namespace VAPoR;

namespace ns_vdfcreate {

int	cvtToExtents(const char *from, void *to);
int	cvtTo3DBool(const char *from, void *to);
int	cvtToOrder(const char *from, void *to);
int	getUserTimes(const char *path, vector <double> &timevec);

struct opt_t {
	OptionParser::Dimension3D_T	dim;
	OptionParser::Dimension3D_T	bs;
	int	numts;
	int	level;
	int	nfilter;
	int	nlifting;
	double deltat;
	double startt;
	char *comment;
	char *coordsystem;
	char *gridtype;
	char *usertimes;
	char *mapprojection;
	char *wname;
	float extents[6];
	int order[3];
	int periodic[3];
	vector <string> varnames;
	vector <string> vars3d;
	vector <string> vars2dxy;
	vector <string> vars2dxz;
	vector <string> vars2dyz;
	vector <int> cratios;
	OptionParser::Boolean_T	vdc2;
	OptionParser::Boolean_T	help;
} opt;

OptionParser::OptDescRec_T	set_opts[] = {
	{"dimension",1, "512x512x512",	"Data volume dimensions expressed in "
		"grid points (NXxNYxNZ)"},
	{"startt", 1, "0.0", "Time, in user coordinates, of the first time step"},
	{"numts",	1, 	"1",			"Number of timesteps in the data set"},
	{"deltat",  1,  "1.0",   "Increment between time steps expressed in user "
		"time coordinates"},
	{"bs",		1, 	"-1x-1x-1",		"Internal storage blocking factor "
		"expressed in grid points (NXxNYxNZ). Defaults: 32x32x32 (VDC type 1), "
		"64x64x64 (VDC type 2"},
	{"level",	1, 	"0",		"Number of approximation levels in hierarchy. "
		"0 => no approximations, 1 => one approximation, and so on (VDC 1 only)"},
	{"nfilter",	1, 	"1","Number of wavelet filter coefficients (VDC 1 only)"},
	{"nlifting",1, 	"1","Number of wavelet lifting coefficients (VDC 1 only)"},
	{"comment",	1,	"",	"Top-level comment to be included in VDF"},
	{"gridtype",	1,	"regular",	"Data grid type "
		"(regular|layered|stretched|block_amr)"}, 
	{"usertimes",	1,	"",	"Path to a file containing a whitespace "
		"delineated list of user times. If present, -numts "
		"option is ignored."}, 
	{"mapprojection",	1,	"",	"A whitespace delineated, quoted list "
        "of PROJ key/value pairs of the form '+paramname=paramvalue'. "
		"vdfcreate does not validate the string for correctness."},
	{"wname",	1,	"bior3.3",	"Wavelet family used for compression "
		"(VDC type 2, only). Recommended values are bior1.1, bior1.3, "
		"bior1.5, bior2.2, bior2.4 ,bior2.6, bior2.8, bior3.1, bior3.3, "
		"bior3.5, bior3.7, bior3.9, bior4.4"},
	{"coordsystem",	1,	"cartesian","Data coordinate system "
		"(cartesian|spherical)"},
	{"extents",	1,	"0:0:0:0:0:0",	"Colon delimited 6-element vector "
		"specifying domain extents in user coordinates (X0:Y0:Z0:X1:Y1:Z1)"},
	{"order",	1,	"0:1:2",	"Colon delimited 3-element vector specifying "
		"permutation ordering of raw data on disk "},
	{"periodic",	1,	"0:0:0",	"Colon delimited 3-element boolean "
		"(0=>nonperiodic, 1=>periodic) vector specifying periodicity of "
		"X,Y,Z coordinate axes (X:Y:Z)"},
	{"varnames",1,	"",				"Deprecated. Use -vars3d instead"},
	{"vars3d",1,	"var1",			"Colon delimited list of 3D variable "
		"names to be included in the VDF"},
	{"vars2dxy",1,	"",			"Colon delimited list of 2D XY-plane variable "
		"names to be included in the VDF"},
	{"vars2dxz",1,	"",			"Colon delimited list of 3D XZ-plane variable "
		"names to be included in the VDF"},
	{"vars2dyz",1,	"",			"Colon delimited list of 3D YZ-plane variable "
		"names to be included in the VDF"},
	{"cratios",1,	"",			"Colon delimited list compression ratios. "
		"The default is 1:10:100:500. " 
		"The maximum compression ratio is wavelet and block size dependent."},
	{"vdc2",	0,	"",	"Generate a VDC Type 2 .vdf file (default is VDC Type 1)"},
	{"help",	0,	"",	"Print this message and exit"},
	{NULL}
};


OptionParser::Option_T	get_options[] = {
	{"dimension", VetsUtil::CvtToDimension3D, &opt.dim, sizeof(opt.dim)},
	{"bs", VetsUtil::CvtToDimension3D, &opt.bs, sizeof(opt.bs)},
	{"numts", VetsUtil::CvtToInt, &opt.numts, sizeof(opt.numts)},
	{"startt", VetsUtil::CvtToDouble, &opt.startt, sizeof(opt.startt)},
	{"deltat", VetsUtil::CvtToDouble, &opt.deltat, sizeof(opt.deltat)},
	{"level", VetsUtil::CvtToInt, &opt.level, sizeof(opt.level)},
	{"nfilter", VetsUtil::CvtToInt, &opt.nfilter, sizeof(opt.nfilter)},
	{"nlifting", VetsUtil::CvtToInt, &opt.nlifting, sizeof(opt.nlifting)},
	{"comment", VetsUtil::CvtToString, &opt.comment, sizeof(opt.comment)},
	{"gridtype", VetsUtil::CvtToString, &opt.gridtype, sizeof(opt.gridtype)},
	{"usertimes", VetsUtil::CvtToString, &opt.usertimes, sizeof(opt.usertimes)},
	{"mapprojection", VetsUtil::CvtToString, &opt.mapprojection, sizeof(opt.mapprojection)},
	{"wname", VetsUtil::CvtToString, &opt.wname, sizeof(opt.wname)},
	{"coordsystem", VetsUtil::CvtToString, &opt.coordsystem, sizeof(opt.coordsystem)},
	{"extents", cvtToExtents, &opt.extents, sizeof(opt.extents)},
	{"order", cvtToOrder, &opt.order, sizeof(opt.order)},
	{"periodic", cvtTo3DBool, &opt.periodic, sizeof(opt.periodic)},
	{"varnames", VetsUtil::CvtToStrVec, &opt.varnames, sizeof(opt.varnames)},
	{"vars3d", VetsUtil::CvtToStrVec, &opt.vars3d, sizeof(opt.vars3d)},
	{"vars2dxy", VetsUtil::CvtToStrVec, &opt.vars2dxy, sizeof(opt.vars2dxy)},
	{"vars2dxz", VetsUtil::CvtToStrVec, &opt.vars2dxz, sizeof(opt.vars2dxz)},
	{"vars2dyz", VetsUtil::CvtToStrVec, &opt.vars2dyz, sizeof(opt.vars2dyz)},
	{"cratios", VetsUtil::CvtToIntVec, &opt.cratios, sizeof(opt.cratios)},
	{"vdc2", VetsUtil::CvtToBoolean, &opt.vdc2, sizeof(opt.vdc2)},
	{"help", VetsUtil::CvtToBoolean, &opt.help, sizeof(opt.help)},
	{NULL}
};

const char *ProgName;

void ErrMsgCBHandler(const char *msg, int) {
    cerr << ProgName << " : " << msg << endl;
}


extern "C" int vdfcreate(int argc, char **argv) {

	OptionParser op;

	MyBase::SetErrMsgCB(ErrMsgCBHandler);

	//
	// Parse command line arguments
	//
	ProgName = Basename(argv[0]);

	size_t bs[3];
	size_t dim[3];
	string	s;
	MetadataVDC *file;


	if (op.AppendOptions(set_opts) < 0) {
		exit(1);
	}

	if (op.ParseOptions(&argc, argv, get_options) < 0) {
		exit(1);
	}

	if (opt.help) {
		cerr << "Usage: " << argv[0] << " [options] filename" << endl;
		op.PrintOptionHelp(stderr);
		exit(0);
	}

	if (argc != 2) {
		cerr << "Usage: " << argv[0] << " [options] filename" << endl;
		op.PrintOptionHelp(stderr);
		exit(1);
	}

	dim[0] = opt.dim.nx;
	dim[1] = opt.dim.ny;
	dim[2] = opt.dim.nz;

	s.assign(opt.coordsystem);

	if (opt.vdc2) {

		if (opt.level) {
			cerr << "The -level option is not supported with VDC2 data" << endl;
			exit(1);
		}

		string wname(opt.wname);
		string wmode;
		if ((wname.compare("bior1.1") == 0) ||
			(wname.compare("bior1.3") == 0) ||
			(wname.compare("bior1.5") == 0) ||
			(wname.compare("bior3.3") == 0) ||
			(wname.compare("bior3.5") == 0) ||
			(wname.compare("bior3.7") == 0) ||
			(wname.compare("bior3.9") == 0)) {

			wmode = "symh"; 
		}
		else if ((wname.compare("bior2.2") == 0) ||
			(wname.compare("bior2.4") == 0) ||
			(wname.compare("bior2.6") == 0) ||
			(wname.compare("bior2.8") == 0) ||
			(wname.compare("bior4.4") == 0)) {

			wmode = "symw"; 
		}
		else {
			wmode = "sp0"; 
		}

		if (opt.bs.nx < 0) {
			for (int i=0; i<3; i++) bs[i] = 64;
		}
		else {
			bs[0] = opt.bs.nx; bs[1] = opt.bs.ny; bs[2] = opt.bs.nz;
		}


		vector <size_t> cratios;
		for (int i=0;i<opt.cratios.size();i++)cratios.push_back(opt.cratios[i]);

		if (cratios.size() == 0) {
			cratios.push_back(1);
			cratios.push_back(10);
			cratios.push_back(100);
			cratios.push_back(500);
		}

		size_t maxcratio = WaveCodecIO::GetMaxCRatio(bs, wname, wmode);
		for (int i=0;i<cratios.size();i++) {
			if (cratios[i] == 0 || cratios[i] > maxcratio) {
				MyBase::SetErrMsg(
					"Invalid compression ratio (%d) for configuration "
					"(block_size, wavename)", cratios[i]
				);
				exit(1);
			}
		}

		file = new MetadataVDC(dim,bs,cratios,wname,wmode);
	}
	else if (s.compare("spherical") == 0) {
		size_t perm[] = {opt.order[0], opt.order[1], opt.order[2]};

		if (opt.bs.nx < 0) {
			for (int i=0; i<3; i++) bs[i] = 32;
		}
		else {
			bs[0] = opt.bs.nx; bs[1] = opt.bs.ny; bs[2] = opt.bs.nz;
		}

		file = new MetadataSpherical(
			dim,opt.level,bs,perm, opt.nfilter,opt.nlifting
		);
	}
	else {
		if (opt.bs.nx < 0) {
			for (int i=0; i<3; i++) bs[i] = 32;
		}
		else {
			bs[0] = opt.bs.nx; bs[1] = opt.bs.ny; bs[2] = opt.bs.nz;
		}
		file = new MetadataVDC(
			dim,opt.level,bs,opt.nfilter,opt.nlifting
		);
	}

	if (MyBase::GetErrCode()) {
		exit(1);
	}

	if (strlen(opt.usertimes)) {
		vector <double> usertimes;
		if (getUserTimes(opt.usertimes, usertimes)<0) {
			exit(1);
		}
		if (file->SetNumTimeSteps(usertimes.size()) < 0) {
			exit(1);
		}
		for (size_t t=0; t<usertimes.size(); t++) {
			vector <double> vec(1,usertimes[t]);
			if (file->SetTSUserTime(t, vec) < 0) {
				exit(1);
			}
		}
	} else {
		if (file->SetNumTimeSteps(opt.numts) < 0) {
			exit(1);
		}
		double usertime = opt.startt;
		for (size_t t=0; t < opt.numts; t++) {
			vector <double> vec(1,usertime);
			if(file->SetTSUserTime(t,vec) < 0) {
				exit(1);
			}
			usertime += opt.deltat;
		}
	}

	if (strlen(opt.mapprojection)) {
		if (file->SetMapProjection(opt.mapprojection) < 0) {
			exit(1);
		}
	}

	s.assign(opt.comment);
	if (file->SetComment(s) < 0) {
		exit(1);
	}

	s.assign(opt.gridtype);
	if (file->SetGridType(s) < 0) {
		exit(1);
	}

	int doExtents = 0;
	for(int i=0; i<5; i++) {
		if (opt.extents[i] != opt.extents[i+1]) doExtents = 1;
	}

	// let Metadata class calculate extents automatically if not 
	// supplied by user explicitly.
	//
	if (doExtents) {
		vector <double> extents;
		for(int i=0; i<6; i++) {
			extents.push_back(opt.extents[i]);
		}
		if (file->SetExtents(extents) < 0) {
			exit(1);
		}
	}

	{
		vector <long> periodic_vec;

		for (int i=0; i<3; i++) periodic_vec.push_back(opt.periodic[i]);

		if (file->SetPeriodicBoundary(periodic_vec) < 0) {
			exit(1);
		}
	}

	// Deal with deprecated option
	if (opt.varnames.size()) opt.vars3d = opt.varnames;

	if (file->GetGridType().compare("layered") == 0){
		//Make sure there's an ELEVATION variable in the vdf
		bool hasElevation = false;
		for (int i = 0; i<opt.vars3d.size(); i++){
			if (opt.vars3d[i].compare("ELEVATION") == 0){
				hasElevation = true;
				break;
			}
		}
		if (!hasElevation){
			opt.vars3d.push_back("ELEVATION");
		}
	}

	if (file->SetVariables3D(opt.vars3d) < 0) {
		exit(1);
	}

	if (file->SetVariables2DXY(opt.vars2dxy) < 0) {
		exit(1);
	}
	if (file->SetVariables2DXZ(opt.vars2dxz) < 0) {
		exit(1);
	}
	if (file->SetVariables2DYZ(opt.vars2dyz) < 0) {
		exit(1);
	}


	if (file->Write(argv[1]) < 0) {
		exit(1);
	}

	
}

int	cvtToOrder(
	const char *from, void *to
) {
	int   *iptr   = (int *) to;

	if (! from) {
		iptr[0] = iptr[1] = iptr[2];
	}
	else if (!  (sscanf(from,"%d:%d:%d", &iptr[0],&iptr[1],&iptr[2]) == 3)) { 

		return(-1);
	}
	return(1);
}

int	cvtToExtents(
	const char *from, void *to
) {
	float   *fptr   = (float *) to;

	if (! from) {
		fptr[0] = fptr[1] = fptr[2] = fptr[3] = fptr[4] = fptr[5] = 0.0;
	}
	else if (! 
		(sscanf(from,"%f:%f:%f:%f:%f:%f",
		&fptr[0],&fptr[1],&fptr[2],&fptr[3],&fptr[4],&fptr[5]) == 6)) { 

		return(-1);
	}
	return(1);
}

int	cvtTo3DBool(
	const char *from, void *to
) {
	int   *iptr   = (int *) to;

	if (! from) {
		iptr[0] = iptr[1] = iptr[2] = 0;
	}
	else if (! (sscanf(from,"%d:%d:%d", &iptr[0],&iptr[1],&iptr[2]) == 3)) { 
		return(-1);
	}
	return(1);
}


int	getUserTimes(const char *path, vector <double> &timevec) {

	ifstream fin(path);
	if (! fin) { 
		MyBase::SetErrMsg("Error opening file %s", path);
		return(-1);
	}

	timevec.clear();

	double d;
	while (fin >> d) {
		timevec.push_back(d);
	}
	fin.close();

	// Make sure times are monotonic.
	//
	int mono = 1;
	if (timevec.size() > 1) {
		if (timevec[0]>=timevec[timevec.size()-1]) {
			for(int i=0; i<timevec.size()-1; i++) {
				if (timevec[i]<timevec[i+1]) mono = 0;
			}
		} else {
			for(int i=0; i<timevec.size()-1; i++) {
				if (timevec[i]>timevec[i+1]) mono = 0;
			}
		}
	}
	if (! mono) {
		MyBase::SetErrMsg("User times sequence must be monotonic");
		return(-1);
	}

	return 0;
}

} // namespace ns_vdfcreate

