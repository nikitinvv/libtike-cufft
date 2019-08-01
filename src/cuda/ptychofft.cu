#include "ptychofft.cuh"
#include "kernels.cuh"
#include<stdio.h>

ptychofft::ptychofft(size_t Ntheta_, size_t Nz_, size_t N_, 
	size_t Nscan_, size_t detx_, size_t dety_, size_t Nprb_)
{
	N = N_;	
	Ntheta = Ntheta_;
	Nz = Nz_;
	Nscan = Nscan_;
	detx = detx_;
	dety = dety_;
	Nprb = Nprb_;

	cudaMalloc((void**)&f,Ntheta*Nz*N*sizeof(float2));
	cudaMalloc((void**)&g,Ntheta*Nscan*detx*dety*sizeof(float2));
	cudaMalloc((void**)&scanx,1*Ntheta*Nscan*sizeof(float));
	cudaMalloc((void**)&scany,1*Ntheta*Nscan*sizeof(float));
	cudaMalloc((void**)&shiftx,1*Ntheta*Nscan*sizeof(float2));
	cudaMalloc((void**)&shifty,1*Ntheta*Nscan*sizeof(float2));
	cudaMalloc((void**)&prb,Ntheta*Nprb*Nprb*sizeof(float2));
	cudaMalloc((void**)&data,Ntheta*Nscan*detx*dety*sizeof(float));	

	int ffts[2];
	int idist;int odist;
	int inembed[2];int onembed[2];
	ffts[0] = detx; ffts[1] = dety;
	idist = detx*dety; odist = detx*dety;
	inembed[0] = detx; inembed[1] = dety;
	onembed[0] = detx; onembed[1] = dety;
	cufftPlanMany(&plan2dfwd, 2, ffts, inembed, 1, idist, onembed, 1, odist, CUFFT_C2C, Ntheta*Nscan); 
}

ptychofft::~ptychofft()
{	
	cudaFree(f);
	cudaFree(g);
	cudaFree(scanx);
	cudaFree(scany);
	cudaFree(shiftx);
	cudaFree(shifty);
	cudaFree(prb);	
	cudaFree(data);	
	cufftDestroy(plan2dfwd);
}

void ptychofft::setobj(size_t scan_, size_t prb_)
{
	cudaMemcpy(scanx,&((float*)scan_)[0],1*Ntheta*Nscan*sizeof(float),cudaMemcpyDefault);  	
	cudaMemcpy(scany,&((float*)scan_)[1*Ntheta*Nscan],1*Ntheta*Nscan*sizeof(float),cudaMemcpyDefault);  	
	cudaMemcpy(prb,(float2*)prb_,Nprb*Nprb*sizeof(float2),cudaMemcpyDefault);
	dim3 BS2d(32,32);
	dim3 GS2d0(ceil(Nscan/(float)BS2d.x),ceil(1*Ntheta/(float)BS2d.y));
	takeshifts<<<GS2d0,BS2d>>>(shiftx,shifty,scanx,scany,1*Ntheta,Nscan);	
}

void ptychofft::fwd(size_t g_, size_t f_, size_t scan_, size_t prb_)
{
	dim3 BS3d(32,32,1);
	dim3 GS2d0(ceil(Nscan/(float)BS3d.x),ceil(Ntheta/(float)BS3d.y));	
	dim3 GS3d0(ceil(Nprb*Nprb/(float)BS3d.x),ceil(Nscan/(float)BS3d.y),ceil(Ntheta/(float)BS3d.z));
	dim3 GS3d1(ceil(detx*dety/(float)BS3d.x),ceil(Nscan/(float)BS3d.y),ceil(Ntheta/(float)BS3d.z));
	
	cudaMemcpy(f,(float2*)f_,Ntheta*Nz*N*sizeof(float2),cudaMemcpyDefault);
	cudaMemset(g,0,Ntheta*Nscan*detx*dety*sizeof(float2));
	cudaMemcpy(scanx,&((float*)scan_)[0],Ntheta*Nscan*sizeof(float),cudaMemcpyDefault);  	
	cudaMemcpy(scany,&((float*)scan_)[Ntheta*Nscan],Ntheta*Nscan*sizeof(float),cudaMemcpyDefault);  	
	cudaMemcpy(prb,(float2*)prb_,Ntheta*Nprb*Nprb*sizeof(float2),cudaMemcpyDefault);  	
		
	takeshifts<<<GS2d0,BS3d>>>(shiftx,shifty,scanx,scany,Ntheta,Nscan);			
	mul<<<GS3d0,BS3d>>>(g,f,prb,scanx,scany,Ntheta,Nz,N,Nscan,Nprb,detx,dety);
	cufftExecC2C(plan2dfwd, (cufftComplex*)g,(cufftComplex*)g,CUFFT_FORWARD);
	shifts<<<GS3d1,BS3d>>>(g, shiftx, shifty, Ntheta, Nscan, detx*dety);
	cudaMemcpy((float2*)g_,g,Ntheta*Nscan*detx*dety*sizeof(float2),cudaMemcpyDefault);  		
}

void ptychofft::adj(size_t f_, size_t g_, size_t scan_, size_t prb_)
{
	dim3 BS3d(32,32,1);
	dim3 GS2d0(ceil(Nscan/(float)BS3d.x),ceil(Ntheta/(float)BS3d.y));	
	dim3 GS3d0(ceil(Nprb*Nprb/(float)BS3d.x),ceil(Nscan/(float)BS3d.y),ceil(Ntheta/(float)BS3d.z));
	dim3 GS3d1(ceil(detx*dety/(float)BS3d.x),ceil(Nscan/(float)BS3d.y),ceil(Ntheta/(float)BS3d.z));

	cudaMemcpy(g,(float2*)g_,Ntheta*Nscan*detx*dety*sizeof(float2),cudaMemcpyDefault);  	
	cudaMemset(f,0,Ntheta*Nz*N*sizeof(float2));	
	cudaMemcpy(scanx,&((float*)scan_)[0],Ntheta*Nscan*sizeof(float),cudaMemcpyDefault);  	
	cudaMemcpy(scany,&((float*)scan_)[Ntheta*Nscan],Ntheta*Nscan*sizeof(float),cudaMemcpyDefault);  	
	cudaMemcpy(prb,(float2*)prb_,Ntheta*Nprb*Nprb*sizeof(float2),cudaMemcpyDefault);  			
	takeshifts<<<GS2d0,BS3d>>>(shiftx,shifty,scanx,scany,Ntheta,Nscan);			

	shiftsa<<<GS3d1,BS3d>>>(g, shiftx, shifty, Ntheta, Nscan, detx*dety);
	cufftExecC2C(plan2dfwd, (cufftComplex*)g,(cufftComplex*)g,CUFFT_INVERSE);
	mula<<<GS3d0,BS3d>>>(f,g,prb,scanx,scany,Ntheta,Nz,N,Nscan,Nprb,detx,dety);
	cudaMemcpy((float2*)f_,f,Ntheta*Nz*N*sizeof(float2),cudaMemcpyDefault);  	
}






