#include "ptychofft.cuh"
#include "kernels.cu"

// constructor, memory allocation
ptychofft::ptychofft(size_t ptheta, size_t nz, size_t n, size_t nscan,
  size_t ndetx, size_t ndety, size_t nprb
) :
  ptheta(ptheta), nz(nz), n(n), nscan(nscan), ndetx(ndetx), ndety(ndety),
  nprb(nprb)
{	
	// create batched 2d FFT plan on GPU with sizes (ndetx,ndety)
	int ffts[2];
	ffts[0] = ndetx;
	ffts[1] = ndety;
	cufftPlanMany(&plan2d, 2, ffts, ffts, 1, ndetx * ndety, ffts, 1, ndetx * ndety, CUFFT_C2C, ptheta * nscan);

	// init 3d thread block on GPU
	BS3d.x = 32;
	BS3d.y = 32;
	BS3d.z = 1;

	// init 3d thread grids	on GPU
	GS3d0.x = ceil(nprb * nprb / (float)BS3d.x);
	GS3d0.y = ceil(nscan / (float)BS3d.y);
	GS3d0.z = ceil(ptheta / (float)BS3d.z);

	GS3d1.x = ceil(ndetx * ndety / (float)BS3d.x);
	GS3d1.y = ceil(nscan / (float)BS3d.y);
	GS3d1.z = ceil(ptheta / (float)BS3d.z);

	GS3d2.x = ceil(nscan / (float)BS3d.x);
	GS3d2.y = ceil(ptheta / (float)BS3d.y);
	GS3d2.z = 1;
}

// destructor, memory deallocation
ptychofft::~ptychofft()
{
  free();
}

void ptychofft::free()
{
  if(!is_free)
  {    
    cufftDestroy(plan2d);
    is_free = true;
  }
}

// forward ptychography operator g = FQf
void ptychofft::fwd(size_t g_, size_t f_, size_t scan_, size_t prb_)
{
  // convert pointers to correct type
  f = (float2 *)f_;
  g = (float2 *)g_;
  scanx = &((float *)scan_)[0];
  scany = &((float *)scan_)[ptheta * nscan];
  prb = (float2 *)prb_;

	// probe multiplication of the object array
	muloperator<<<GS3d0, BS3d>>>(f, g, prb, scanx, scany, ptheta, nz, n, nscan, nprb, ndetx, ndety, 2); //flg==2 forward transform
	// Fourier transform
	cufftExecC2C(plan2d, (cufftComplex *)g, (cufftComplex *)g, CUFFT_FORWARD);
}

// adjoint ptychography operator with respect to object (flg==0) f = Q*F*g, or probe (flg==1) prb = Q*F*g
void ptychofft::adj(size_t f_, size_t g_, size_t scan_, size_t prb_, int flg)
{
  // convert pointers to correct type
  f = (float2 *)f_;
  g = (float2 *)g_;
  scanx = &((float *)scan_)[0];
  scany = &((float *)scan_)[ptheta * nscan];
  prb = (float2 *)prb_;

	// inverse Fourier transform
	cufftExecC2C(plan2d, (cufftComplex *)g, (cufftComplex *)g, CUFFT_INVERSE);	
	// adjoint probe (flg==0) or object (flg=1) multiplication operator
	muloperator<<<GS3d0, BS3d>>>(f, g, prb, scanx, scany, ptheta, nz, n, nscan, nprb, ndetx, ndety, flg);	
}