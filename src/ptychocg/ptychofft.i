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
	size_t N;	  // object horizontal size
	size_t Nz;	 // object vertical size
	size_t Ntheta; // number of projections
	size_t Nscan;  // number of scan positions for 1 projection
	size_t Ndetx;  // detector x size
	size_t Ndety;  // detector y size
	size_t Nprb;   // probe size in 1 dimension

	float2 *f;		// object
	float2 *g;		// data
	float2 *prb;	// probe function
	float *scanx;   // x scan positions
	float *scany;   // y scan positions
	float2 *shiftx; // x shift (-1,1) of scan positions to nearest integer
	float2 *shifty; // y shift (-1,1) of scan positions to nearest integer

	cufftHandle plan2d; // 2D FFT plan
	cufftHandle plan2dshift; // 2D FFT plan for the shift in the frequency domain

	dim3 BS3d; // 3d thread block on GPU

	// 3d thread grids on GPU for different kernels
	dim3 GS3d0;
	dim3 GS3d1;
	dim3 GS3d2;

public:
	// constructor, memory allocation
	ptychofft(size_t Ntheta, size_t Nz, size_t N,
			  size_t Nscan, size_t Ndetx, size_t Ndety, size_t Nprb);
	// destructor, memory deallocation
	~ptychofft();
	// forward ptychography operator FQ
	void fwd(size_t g_, size_t f_, size_t scan_, size_t prb_);
	// adjoint ptychography operator with respect to object (fgl==0) f = Q*F*g, or probe (flg==1) prb = Q*F*g
	void adj(size_t f_, size_t g_, size_t scan_, size_t prb_, int flg);
	// adjoint ptychography operator Q*F* with respect to probe	
};
