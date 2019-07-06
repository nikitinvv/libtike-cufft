/*interface*/
%module ptychofft

%{
#define SWIG_FILE_WITH_INIT
#include "ptychofft.cuh"
%}

%include "numpy.i"

%init %{
import_array();
%}
class ptychofft
{
	size_t N;
	size_t Ntheta;
	size_t Nz;
	size_t Nscan;
	size_t detx;
	size_t dety;
	size_t Nprb;

	float2* f;
	float2* g;
	float2* prb; 
	float* scanx; 
	float* scany; 
	float2* shiftx; 
	float2* shifty; 
	float2* ff;
	float2* fff;
	float* data;
	float2* ftmp0;
	float2* ftmp1;
	
	cufftHandle plan2dfwd;
	cufftHandle plan2dadj;

public:
	ptychofft(size_t Ntheta, size_t Nz, size_t N, size_t Ntheta0,
		size_t Nscan, size_t detx, size_t dety, size_t Nprb);
	~ptychofft();	
	void setobj(size_t scan_,  size_t prb_);
	void fwd(size_t g_, size_t f_);
	void adj(size_t f_, size_t g_);	
	
};


