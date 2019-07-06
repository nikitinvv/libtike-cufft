#include "radonusfft.cuh"
#include "kernels.cuh"
#include <stdio.h>

radonusfft::radonusfft(size_t Ntheta_, size_t Nz_, size_t N_)
{
	N = N_*3/2;
	Ntheta = Ntheta_;
	Nz = Nz_;
	float eps = 1e-6;
	mu = -log(eps)/(2*N*N);
	M = ceil(2*N*1/PI*sqrt(-mu*log(eps)+(mu*N)*(mu*N)/4));

	cudaMalloc((void**)&f,N*N*Nz*sizeof(float2));
	cudaMalloc((void**)&g,N*Ntheta*Nz*sizeof(float2));
	cudaMalloc((void**)&fde,2*N*2*N*Nz*sizeof(float2));
	cudaMalloc((void**)&fdee,(2*N+2*M)*(2*N+2*M)*Nz*sizeof(float2));

	cudaMalloc((void**)&x,N*Ntheta*sizeof(float));
	cudaMalloc((void**)&y,N*Ntheta*sizeof(float));
	cudaMalloc((void**)&theta,Ntheta*sizeof(float));

	int ffts[2];
	int idist;int odist;
	int inembed[2];int onembed[2];
	//fft 2d 
	ffts[0] = 2*N; ffts[1] = 2*N;
	idist = 2*N*2*N;odist = (2*N+2*M)*(2*N+2*M);
	inembed[0] = 2*N; inembed[1] = 2*N;
	onembed[0] = 2*N+2*M; onembed[1] = 2*N+2*M;
	cufftPlanMany(&plan2dfwd, 2, ffts, inembed, 1, idist, onembed, 1, odist, CUFFT_C2C, Nz); 
	cufftPlanMany(&plan2dadj, 2, ffts, onembed, 1, odist, inembed, 1, idist, CUFFT_C2C, Nz); 

	//fft 1d	
	ffts[0] = N;
	idist = N;odist = N;
	inembed[0] = N;onembed[0] = N;
	cufftPlanMany(&plan1d, 1, ffts, inembed, 1, idist, onembed, 1, odist, CUFFT_C2C, Ntheta*Nz);
}

radonusfft::~radonusfft()
{	
	cudaFree(f);
	cudaFree(g);			
	cudaFree(fde);
	cudaFree(fdee);
	cudaFree(x);
	cudaFree(y);
	cufftDestroy(plan2dfwd);
	cufftDestroy(plan2dadj);
	cufftDestroy(plan1d);
}

void radonusfft::fwd(size_t g_, size_t f_)
{	
	dim3 BS2d(32,32);
	dim3 BS3d(32,32,1);

	dim3 GS2d0(ceil(N/(float)BS2d.x),ceil(Ntheta/(float)BS2d.y));
	dim3 GS3d0(ceil(N/(float)BS3d.x),ceil(N/(float)BS3d.y),ceil(Nz/(float)BS3d.z));
	dim3 GS3d1(ceil(2*N/(float)BS3d.x),ceil(2*N/(float)BS3d.y),ceil(Nz/(float)BS3d.z));
	dim3 GS3d2(ceil((2*N+2*M)/(float)BS3d.x),ceil((2*N+2*M)/(float)BS3d.y),ceil(Nz/(float)BS3d.z));
	dim3 GS3d3(ceil(N/(float)BS3d.x),ceil(Ntheta/(float)BS3d.y),ceil(Nz/(float)BS3d.z));

	//cudaMemcpy(f,(float2*)f_,N*N*Nz*sizeof(float2),cudaMemcpyDefault);
	//padded version
	cudaMemset(f,0,N*N*Nz*sizeof(float2));
	for(int iz=0;iz<Nz;iz++)
		cudaMemcpy2D(&f[iz*N*N+N/6*N+N/6],N*sizeof(float2),&((float2*)f_)[iz*N/3*2*N/3*2],N/3*2*sizeof(float2),N/3*2*sizeof(float2),N/3*2*1,cudaMemcpyDefault);

	cudaMemset(fde,0,2*N*2*N*Nz*sizeof(float2));
	cudaMemset(fdee,0,(2*N+2*M)*(2*N+2*M)*Nz*sizeof(float2));

	pad2<<<GS3d0, BS3d>>>(f,N,Nz);
	circ<<<GS3d0, BS3d>>>(f,1.0f/N,N,Nz);
	takexy<<<GS2d0, BS2d>>>(x,y,theta,N,Ntheta);

	divphi<<<GS3d0, BS3d>>>(fde,f,mu,N,Nz);
	fftshiftc<<<GS3d1, BS3d>>>(fde,2*N,Nz);
	cufftExecC2C(plan2dfwd, (cufftComplex*)fde,(cufftComplex*)&fdee[M+M*(2*N+2*M)],CUFFT_FORWARD);
	fftshiftc<<<GS3d2, BS3d>>>(fdee,2*N+2*M,Nz);

	wrap<<<GS3d2, BS3d>>>(fdee,N,Nz,M);
	gather<<<GS3d3, BS3d>>>(g,fdee,x,y,M,mu,N,Ntheta,Nz);

	fftshift1c<<<GS3d3, BS3d>>>(g,N,Ntheta,Nz);
	cufftExecC2C(plan1d, (cufftComplex*)g,(cufftComplex*)g,CUFFT_INVERSE);
	fftshift1c<<<GS3d3, BS3d>>>(g,N,Ntheta,Nz);

	//cudaMemcpy((float2*)g_,g,N*Ntheta*Nz*sizeof(float2),cudaMemcpyDefault);
	cudaMemcpy2D((float2*)g_,N/3*2*sizeof(float2),&g[N/6],N*sizeof(float2),N/3*2*sizeof(float2),Ntheta*Nz,cudaMemcpyDefault);  	
}

void radonusfft::adj(size_t f_, size_t g_)
{
	dim3 BS2d(32,32);
	dim3 BS3d(32,32,1);

	dim3 GS2d0(ceil(N/(float)BS2d.x),ceil(Ntheta/(float)BS2d.y));
	dim3 GS3d0(ceil(N/(float)BS3d.x),ceil(N/(float)BS3d.y),ceil(Nz/(float)BS3d.z));
	dim3 GS3d1(ceil(2*N/(float)BS3d.x),ceil(2*N/(float)BS3d.y),ceil(Nz/(float)BS3d.z));
	dim3 GS3d2(ceil((2*N+2*M)/(float)BS3d.x),ceil((2*N+2*M)/(float)BS3d.y),ceil(Nz/(float)BS3d.z));
	dim3 GS3d3(ceil(N/(float)BS3d.x),ceil(Ntheta/(float)BS3d.y),ceil(Nz/(float)BS3d.z));

	//cudaMemcpy(g,(float2*)g_,N*Ntheta*Nz*sizeof(float2),cudaMemcpyDefault);
	cudaMemset(g,0,N*Ntheta*Nz*sizeof(float2));
	cudaMemcpy2D(&g[N/6],N*sizeof(float2),(float2*)g_,N/3*2*sizeof(float2),N/3*2*sizeof(float2),Ntheta*Nz,cudaMemcpyDefault);  	


	cudaMemset(fde,0,(2*N+2*M)*(2*N+2*M)*Nz*sizeof(float2));
	cudaMemset(fdee,0,(2*N+2*M)*(2*N+2*M)*Nz*sizeof(float2));


	//padded version
	pad<<<GS3d3,BS3d>>>(g,N,Ntheta,Nz);
	takexy<<<GS2d0, BS2d>>>(x,y,theta,N,Ntheta);

	fftshift1c<<<GS3d3, BS3d>>>(g,N,Ntheta,Nz);
	cufftExecC2C(plan1d, (cufftComplex*)g,(cufftComplex*)g,CUFFT_FORWARD);
	fftshift1c<<<GS3d3, BS3d>>>(g,N,Ntheta,Nz);
	//applyfilter<<<GS3d3, BS3d>>>(g,N,Ntheta,Nz);


	scatter<<<GS3d3, BS3d>>>(fdee,g,x,y,M,mu,N,Ntheta,Nz);
	wrapadj<<<GS3d2, BS3d>>>(fdee,N,Nz,M);

	fftshiftc<<<GS3d2, BS3d>>>(fdee,2*N+2*M,Nz);
	cufftExecC2C(plan2dadj, (cufftComplex*)&fdee[M+M*(2*N+2*M)],(cufftComplex*)fde,CUFFT_INVERSE);
	fftshiftc<<<GS3d1, BS3d>>>(fde,2*N,Nz);

	unpaddivphi<<<GS3d0, BS3d>>>(f,fde,mu,N,Nz);
	circ<<<GS3d0, BS3d>>>(f,1.0f/N,N,Nz);

	//cudaMemcpy((float2*)f_,f,N*N*Nz*sizeof(float2),cudaMemcpyDefault);  	
	//padded version
	for(int iz=0;iz<Nz;iz++)
		cudaMemcpy2D(&((float2*)f_)[iz*N/3*2*N/3*2],N/3*2*sizeof(float2),&f[iz*N*N+N/6*N+N/6],N*sizeof(float2),N/3*2*sizeof(float2),N/3*2*1,cudaMemcpyDefault);

}

void radonusfft::setobj(size_t theta_)
{
	cudaMemcpy(theta,(float*)theta_,Ntheta*sizeof(float),cudaMemcpyDefault);  	
}
