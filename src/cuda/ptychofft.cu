#include "ptychofft.cuh"
#include "kernels.cuh"

// constructor, memory allocation
ptychofft::ptychofft(size_t Ntheta_, size_t Nz_, size_t N_,
					 size_t Nscan_, size_t Ndetx_, size_t Ndety_, size_t Nprb_)
{
	// init sizes
	N = N_;
	Ntheta = Ntheta_;
	Nz = Nz_;
	Nscan = Nscan_;
	Ndetx = Ndetx_;
	Ndety = Ndety_;
	Nprb = Nprb_;

	// allocate memory on GPU
	cudaMalloc((void **)&f, Ntheta * Nz * N * sizeof(float2));
	cudaMalloc((void **)&g, Ntheta * Nscan * Ndetx * Ndety * sizeof(float2));
	cudaMalloc((void **)&scanx, Ntheta * Nscan * sizeof(float));
	cudaMalloc((void **)&scany, Ntheta * Nscan * sizeof(float));
	cudaMalloc((void **)&shiftx, Ntheta * Nscan * sizeof(float2));
	cudaMalloc((void **)&shifty, Ntheta * Nscan * sizeof(float2));
	cudaMalloc((void **)&prb, Ntheta * Nprb * Nprb * sizeof(float2));

	// create batched 2d FFT plan on GPU
	int ffts[2];
	ffts[0] = Ndetx;
	ffts[1] = Ndety;	
	cufftPlanMany(&plan2d, 2, ffts, ffts, 1, Ndetx * Ndety, ffts, 1, Ndetx * Ndety, CUFFT_C2C, Ntheta * Nscan);

	// init 3d thread block on GPU
	BS3d.x = 32;
	BS3d.y = 32;
	BS3d.z = 1;

	// init 3d thread grids	on GPU
	GS3d0.x = ceil(Nprb * Nprb / (float)BS3d.x);
	GS3d0.y = ceil(Nscan / (float)BS3d.y);
	GS3d0.z = ceil(Ntheta / (float)BS3d.z);

	GS3d1.x = ceil(Ndetx * Ndety / (float)BS3d.x);
	GS3d1.y = ceil(Nscan / (float)BS3d.y);
	GS3d1.z = ceil(Ntheta / (float)BS3d.z);

	GS3d2.x = ceil(Nscan / (float)BS3d.x);
	GS3d2.y = ceil(Ntheta / (float)BS3d.y);
	GS3d2.z = 1;
}

// destructor, memory deallocation
ptychofft::~ptychofft()
{
	cudaFree(f);
	cudaFree(g);
	cudaFree(scanx);
	cudaFree(scany);
	cudaFree(shiftx);
	cudaFree(shifty);
	cudaFree(prb);
	cufftDestroy(plan2d);
}

// forward ptychography operator g = FQf
void ptychofft::fwd(size_t g_, size_t f_, size_t scan_, size_t prb_)
{
	// copy arrays to GPU
	cudaMemcpy(f, (float2 *)f_, Ntheta * Nz * N * sizeof(float2), cudaMemcpyDefault);
	cudaMemset(g, 0, Ntheta * Nscan * Ndetx * Ndety * sizeof(float2));
	cudaMemcpy(scanx, &((float *)scan_)[0], Ntheta * Nscan * sizeof(float), cudaMemcpyDefault);
	cudaMemcpy(scany, &((float *)scan_)[Ntheta * Nscan], Ntheta * Nscan * sizeof(float), cudaMemcpyDefault);
	cudaMemcpy(prb, (float2 *)prb_, Ntheta * Nprb * Nprb * sizeof(float2), cudaMemcpyDefault);

	// probe multiplication of the object array
	mulprobe<<<GS3d0, BS3d>>>(g, f, prb, scanx, scany, Ntheta, Nz, N, Nscan, Nprb, Ndetx, Ndety);
	// Fourier transform
	cufftExecC2C(plan2d, (cufftComplex *)g, (cufftComplex *)g, CUFFT_FORWARD);
	// compute exp(1j dx),exp(1j dy) where dx,dy are in (-1,1) and correspond to shifts to nearest integer
	takeshifts<<<GS3d2, BS3d>>>(shiftx, shifty, scanx, scany, Ntheta, Nscan);
	// perform shifts in the frequency domain by multiplication with exp(1j dx),exp(1j dy)
	shifts<<<GS3d1, BS3d>>>(g, shiftx, shifty, 1, Ntheta, Nscan, Ndetx * Ndety);

	// copy result to CPU
	cudaMemcpy((float2 *)g_, g, Ntheta * Nscan * Ndetx * Ndety * sizeof(float2), cudaMemcpyDefault);
}

// adjoint ptychography operator with respect to object (fgl==0) f = Q*F*g, or probe (flg==1) prb = Q*F*g
void ptychofft::adj(size_t f_, size_t g_, size_t scan_, size_t prb_, int flg)
{
	// copy arrays to GPU
	cudaMemcpy(g, (float2 *)g_, Ntheta * Nscan * Ndetx * Ndety * sizeof(float2), cudaMemcpyDefault);
	cudaMemcpy(scanx, &((float *)scan_)[0], Ntheta * Nscan * sizeof(float), cudaMemcpyDefault);
	cudaMemcpy(scany, &((float *)scan_)[Ntheta * Nscan], Ntheta * Nscan * sizeof(float), cudaMemcpyDefault);
	cudaMemcpy(prb, (float2 *)prb_, Ntheta * Nprb * Nprb * sizeof(float2), cudaMemcpyDefault);

	// init object as 0
	cudaMemset(f, 0, Ntheta * Nz * N * sizeof(float2));
	// compute exp(1j dx),exp(1j dy) where dx,dy are in (-1,1) and correspond to shifts to nearest integer
	takeshifts<<<GS3d2, BS3d>>>(shiftx, shifty, scanx, scany, Ntheta, Nscan);
	// perform shifts in the frequency domain by multiplication with exp(-1j dx),exp(-1j dy) - backward
	shifts<<<GS3d1, BS3d>>>(g, shiftx, shifty, -1, Ntheta, Nscan, Ndetx * Ndety);
	// inverse Fourier transform
	cufftExecC2C(plan2d, (cufftComplex *)g, (cufftComplex *)g, CUFFT_INVERSE);
	if (flg == 0)
	{
		// adjoint probe multiplication and simultaneous writing to the object array
		mulaprobe<<<GS3d0, BS3d>>>(f, g, prb, scanx, scany, Ntheta, Nz, N, Nscan, Nprb, Ndetx, Ndety);
		// copy result to CPU
		cudaMemcpy((float2 *)f_, f, Ntheta * Nz * N * sizeof(float2), cudaMemcpyDefault);
	}
	else if (flg == 1)
	{
		// adjoint probe multiplication and simultaneous writing to the probe array
		mulaprobeq<<<GS3d0, BS3d>>>(prb, g, f, scanx, scany, Ntheta, Nz, N, Nscan, Nprb, Ndetx, Ndety);
		// copy result to CPU
		cudaMemcpy((float2 *)f_, prb, Ntheta * Nprb * Nprb * sizeof(float2), cudaMemcpyDefault);
	}
}
