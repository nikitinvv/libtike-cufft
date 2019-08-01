#define PI 3.1415926535

// Tomography kernels

void __global__ divphi(float2 *g, float2 *f, float mu, int N, int Nz)
{
	int tx = blockDim.x * blockIdx.x + threadIdx.x;
	int ty = blockDim.y * blockIdx.y + threadIdx.y;
	int tz = blockDim.z * blockIdx.z + threadIdx.z;

	if (tx>=N||ty>=N||tz>=Nz) return;
	float phi = __expf(-mu*(tx-N/2)*(tx-N/2)-mu*(ty-N/2)*(ty-N/2));
	g[tx+N/2+(ty+N/2)*2*N+tz*4*N*N].x = f[tx+ty*N+tz*N*N].x/phi/(4*N*N);
	g[tx+N/2+(ty+N/2)*2*N+tz*4*N*N].y = f[tx+ty*N+tz*N*N].y/phi/(4*N*N);
}

void __global__ unpaddivphi(float2 *f, float2 *g,float mu, int N, int Nz)
{
	int tx = blockDim.x * blockIdx.x + threadIdx.x;
	int ty = blockDim.y * blockIdx.y + threadIdx.y;
	int tz = blockDim.z * blockIdx.z + threadIdx.z;
	if (tx>=N||ty>=N||tz>=Nz) return;
	float phi=__expf(-mu*(tx-N/2)*(tx-N/2)-mu*(ty-N/2)*(ty-N/2));
	f[tx+ty*N+tz*N*N].x = g[tx+N/2+(ty+N/2)*2*N+tz*4*N*N].x/phi/(4*N*N);
	f[tx+ty*N+tz*N*N].y = g[tx+N/2+(ty+N/2)*2*N+tz*4*N*N].y/phi/(4*N*N);
}

void __global__ fftshiftc(float2 *f, int N, int Nz)
{
	int tx = blockDim.x * blockIdx.x + threadIdx.x;
	int ty = blockDim.y * blockIdx.y + threadIdx.y;
	int tz = blockDim.z * blockIdx.z + threadIdx.z;
	if (tx>=N||ty>=N||tz>=Nz) return;
	int g = (1-2*((tx+1)%2))*(1-2*((ty+1)%2));
	f[tx+ty*N+tz*N*N].x *= g;
	f[tx+ty*N+tz*N*N].y *= g;
}

void __global__ fftshift1c(float2 *f, int N, int Ntheta, int Nz)
{
	int tx = blockDim.x * blockIdx.x + threadIdx.x;
	int ty = blockDim.y * blockIdx.y + threadIdx.y;
	int tz = blockDim.z * blockIdx.z + threadIdx.z;
	if (tx>=N||ty>=Ntheta||tz>=Nz) return;
	int g = (1-2*((tx+1)%2));
	f[tx+tz*N+ty*N*Nz].x *= g;
	f[tx+tz*N+ty*N*Nz].y *= g;
}

void __global__ wrap(float2 *f, int N, int Nz, int M)
{
	int tx = blockDim.x * blockIdx.x + threadIdx.x;
	int ty = blockDim.y * blockIdx.y + threadIdx.y;
	int tz = blockDim.z * blockIdx.z + threadIdx.z;
	if (tx>=2*N+2*M||ty>=2*N+2*M||tz>=Nz) return;
	if (tx<M||tx>=2*N+M||ty<M||ty>=2*N+M)
	{
		int tx0 = (tx-M+2*N)%(2*N);
		int ty0 = (ty-M+2*N)%(2*N);
		int id1 = tx+ty*(2*N+2*M)+tz*(2*N+2*M)*(2*N+2*M);
		int id2 = tx0+M+(ty0+M)*(2*N+2*M)+tz*(2*N+2*M)*(2*N+2*M);
		f[id1].x = f[id2].x;
		f[id1].y = f[id2].y;
	}
}

void __global__ wrapadj(float2 *f, int N, int Nz, int M)
{
	int tx = blockDim.x * blockIdx.x + threadIdx.x;
	int ty = blockDim.y * blockIdx.y + threadIdx.y;
	int tz = blockDim.z * blockIdx.z + threadIdx.z;
	if (tx>=2*N+2*M||ty>=2*N+2*M||tz>=Nz) return;
	if (tx<M||tx>=2*N+M||ty<M||ty>=2*N+M)
	{
		int tx0 = (tx-M+2*N)%(2*N);
		int ty0 = (ty-M+2*N)%(2*N);
		int id1 = tx+ty*(2*N+2*M)+tz*(2*N+2*M)*(2*N+2*M);
		int id2 = tx0+M+(ty0+M)*(2*N+2*M)+tz*(2*N+2*M)*(2*N+2*M);
		atomicAdd(&f[id2].x,f[id1].x);
		atomicAdd(&f[id2].y,f[id1].y);
	}
}

void __global__ takexy(float *x, float *y, float *theta, int N, int Ntheta)
{
        int tx = blockDim.x * blockIdx.x + threadIdx.x;
        int ty = blockDim.y * blockIdx.y + threadIdx.y;

        if (tx>=N||ty>=Ntheta) return;
        x[tx+ty*N] = (tx-N/2)/(float)N*__cosf(theta[ty]);
        y[tx+ty*N] = -(tx-N/2)/(float)N*__sinf(theta[ty]);
	if (x[tx+ty*N]>=0.5f) x[tx+ty*N]=0.5f-1e-5;
	if (y[tx+ty*N]>=0.5f) y[tx+ty*N]=0.5f-1e-5;

}

void __global__ gather(float2* g,float2 *f,float *x,float *y, int M, float mu,int N,int Ntheta, int Nz)
{
	int tx = blockDim.x * blockIdx.x + threadIdx.x;
	int ty = blockDim.y * blockIdx.y + threadIdx.y;
	int tz = blockDim.z * blockIdx.z + threadIdx.z;

	if (tx>=N||ty>=Ntheta||tz>=Nz) return;

	float x0,y0;
	float2 g0;
	x0 = x[tx+ty*N];
	y0 = y[tx+ty*N];
	g0.x = 0.0f;g0.y = 0.0f;

	for (int i1=0;i1<2*M+1;i1++)
	{
		int ell1 = floorf(2*N*y0)-M+i1;
		for (int i0=0;i0<2*M+1;i0++)
		{
			int ell0 = floorf(2*N*x0)-M+i0;
			float w0 = ell0/(float)(2*N)-x0;
			float w1 = ell1/(float)(2*N)-y0;
			float w = PI/(sqrtf(mu*mu))*__expf(-PI*PI/mu*(w0*w0)-PI*PI/mu*(w1*w1));
			g0.x += w*f[N+M+ell0+(2*N+2*M)*(N+M+ell1)+tz*(2*N+2*M)*(2*N+2*M)].x;
			g0.y += w*f[N+M+ell0+(2*N+2*M)*(N+M+ell1)+tz*(2*N+2*M)*(2*N+2*M)].y;
		}
	}
	g[tx+tz*N+ty*N*Nz].x = g0.x/N;
	g[tx+tz*N+ty*N*Nz].y = g0.y/N;
}

void __global__ scatter(float2* f,float2 *g,float *x,float *y, int M, float mu,int N,int Ntheta, int Nz)
{
	int tx = blockDim.x * blockIdx.x + threadIdx.x;
	int ty = blockDim.y * blockIdx.y + threadIdx.y;
	int tz = blockDim.z * blockIdx.z + threadIdx.z;

	if (tx>=N||ty>=Ntheta||tz>=Nz) return;

	float x0,y0;
	float2 g0;
	x0 = x[tx+ty*N];
	y0 = y[tx+ty*N];
	g0.x = g[tx+tz*N+ty*N*Nz].x/N;
	g0.y = g[tx+tz*N+ty*N*Nz].y/N;

	for (int i1=0;i1<2*M+1;i1++)
	{
		int ell1=floorf(2*N*y0)-M+i1;
		for (int i0=0;i0<2*M+1;i0++)
		{
			int ell0=floorf(2*N*x0)-M+i0;
			float w0=ell0/(float)(2*N)-x0;
			float w1=ell1/(float)(2*N)-y0;
			float w=PI/(sqrtf(mu*mu))*__expf(-PI*PI/mu*(w0*w0)-PI*PI/mu*(w1*w1));
			float* fx=&(f[N+M+ell0+(2*N+2*M)*(N+M+ell1)+tz*(2*N+2*M)*(2*N+2*M)].x);
			float* fy=&(f[N+M+ell0+(2*N+2*M)*(N+M+ell1)+tz*(2*N+2*M)*(2*N+2*M)].y);
			atomicAdd(fx,w*g0.x);
			atomicAdd(fy,w*g0.y);
		}
	}
}

void __global__ circ(float2 *f, float r, int N, int Nz)
{
	int tx = blockDim.x * blockIdx.x + threadIdx.x;
	int ty = blockDim.y * blockIdx.y + threadIdx.y;
	int tz = blockDim.z * blockIdx.z + threadIdx.z;
	if (tx>=N||ty>=N||tz>=Nz) return;
	int id0 = tx+ty*N+tz*N*N;
	float x = (tx-N/2)/float(N);
	float y = (ty-N/2)/float(N);
	int lam = (4*x*x+4*y*y)<1-r;
	f[id0].x *= lam;
	f[id0].y *= lam;
}

void __global__ applyfilter(float2 *f, int N, int Ntheta, int Nz)
{
	int tx = blockDim.x * blockIdx.x + threadIdx.x;
	int ty = blockDim.y * blockIdx.y + threadIdx.y;
	int tz = blockDim.z * blockIdx.z + threadIdx.z;
	if (tx>=N||ty>=Ntheta||tz>=Nz) return;
	int id0 = tx+ty*N+tz*Ntheta*N;
	float rho=(tx-N/2)/(float)N;
	float w=0;
	if(rho==0) w=0;
	else w=abs(rho)*N*4*sin(rho)/rho;//(1-fabs(rho)/coef)*(1-fabs(rho)/coef)*(1-fabs(rho)/coef);
	f[id0].x*=w;
	f[id0].y*=w;
}

void __global__ pad(float2 *f, int N, int Ntheta, int Nz)
{
	int tx = blockDim.x * blockIdx.x + threadIdx.x;
	int ty = blockDim.y * blockIdx.y + threadIdx.y;
	int tz = blockDim.z * blockIdx.z + threadIdx.z;
	if (tx>=N||ty>=Ntheta||tz>=Nz) return;
	if(tx<N/6)
	{
		f[tx+tz*N+ty*N*Nz].x = f[N/6+tz*N+ty*N*Nz].x;
		f[tx+tz*N+ty*N*Nz].y = f[N/6+tz*N+ty*N*Nz].y;
	}
	if(tx>=5*N/6)
	{
		f[tx+tz*N+ty*N*Nz].x = f[5*N/6-1+tz*N+ty*N*Nz].x;
		f[tx+tz*N+ty*N*Nz].y = f[5*N/6-1+tz*N+ty*N*Nz].y;
	}

}

void __global__ pad2(float2 *f, int N, int Nz)
{
	int tx = blockDim.x * blockIdx.x + threadIdx.x;
	int ty = blockDim.y * blockIdx.y + threadIdx.y;
	int tz = blockDim.z * blockIdx.z + threadIdx.z;
	if (tx>=N||ty>=N||tz>=Nz) return;
	f[tx+ty*N+tz*N*N].x = f[max(N/6,min(5*N/6-1,tx))+max(N/6,min(5*N/6-1,ty))*N+tz*N*N].x;
	f[tx+ty*N+tz*N*N].y = f[max(N/6,min(5*N/6-1,tx))+max(N/6,min(5*N/6-1,ty))*N+tz*N*N].y;
}

void __global__ subdata(float2 *f, float2* data, int N, int Ntheta, int Nz)
{
	int tx = blockDim.x * blockIdx.x + threadIdx.x;
	int ty = blockDim.y * blockIdx.y + threadIdx.y;
	int tz = blockDim.z * blockIdx.z + threadIdx.z;
	if (tx>=N||ty>=Ntheta||tz>=Nz) return;

	f[tx+tz*N+ty*N*Nz].x -= data[tx+tz*N+ty*N*Nz].x;
	f[tx+tz*N+ty*N*Nz].y -= data[tx+tz*N+ty*N*Nz].y;
}

void __global__ updatef(float2 *f, float2* ff, float eta, int N, int Nz)
{
	int tx = blockDim.x * blockIdx.x + threadIdx.x;
	int ty = blockDim.y * blockIdx.y + threadIdx.y;
	int tz = blockDim.z * blockIdx.z + threadIdx.z;
	if (tx>=N||ty>=N||tz>=Nz) return;
	int id0 = tx+ty*N+tz*N*N;
	f[id0].x-=2*eta*ff[id0].x;
	f[id0].y-=2*eta*ff[id0].y;
}


// Pytchography kernels

void __global__ mul(float2 *g, float2 *f, float2 *prb, float *scanx, float *scany,
	int Ntheta, int Nz, int N, int Nscan, int Nprb, int detx, int dety)
{
	int tx = blockDim.x * blockIdx.x + threadIdx.x;
	int ty = blockDim.y * blockIdx.y + threadIdx.y;
	int tz = blockDim.z * blockIdx.z + threadIdx.z;

	if (tx>=Nprb*Nprb||ty>=Nscan||tz>=Ntheta) return;
	int ix = tx%Nprb;
	int iy = tx/Nprb;

	int stx = roundf(scanx[ty+tz*Nscan]);
	int sty = roundf(scany[ty+tz*Nscan]);
	if(stx<0||sty<0) return;

	int shift = (dety-Nprb)/2*detx+(detx-Nprb)/2;
	float2 f0 = f[(stx+ix)+(sty+iy)*N+tz*Nz*N];
	float2 prb0 = prb[ix+iy*Nprb+tz*Nprb*Nprb];
	float c = 1/sqrtf(detx*dety);//fft constant
	g[shift+ix+iy*detx+ty*detx*dety+tz*detx*dety*Nscan].x = c*prb0.x*f0.x-c*prb0.y*f0.y;
	g[shift+ix+iy*detx+ty*detx*dety+tz*detx*dety*Nscan].y = c*prb0.x*f0.y+c*prb0.y*f0.x;

}

void __global__ mula(float2 *f, float2 *g, float2 *prb, float *scanx, float *scany,
	int Ntheta, int Nz, int N, int Nscan, int Nprb, int detx, int dety)
{
	int tx = blockDim.x * blockIdx.x + threadIdx.x;
	int ty = blockDim.y * blockIdx.y + threadIdx.y;
	int tz = blockDim.z * blockIdx.z + threadIdx.z;

	if (tx>=Nprb*Nprb||ty>=Nscan||tz>=Ntheta) return;
	int ix = tx%Nprb;
	int iy = tx/Nprb;

	int stx = roundf(scanx[ty+tz*Nscan]);
	int sty = roundf(scany[ty+tz*Nscan]);
	if(stx<0||sty<0) return;

	int shift = (dety-Nprb)/2*detx+(detx-Nprb)/2;
	float2 g0 = g[shift+ix+iy*detx+ty*detx*dety+tz*detx*dety*Nscan];
	float2 prb0 = prb[ix+iy*Nprb+tz*Nprb*Nprb];
	float c = 1/sqrtf(detx*dety);//fft constant
	atomicAdd(&f[(stx+ix)+(sty+iy)*N+tz*Nz*N].x, c*prb0.x*g0.x+c*prb0.y*g0.y);
	atomicAdd(&f[(stx+ix)+(sty+iy)*N+tz*Nz*N].y, c*prb0.x*g0.y-c*prb0.y*g0.x);
}

void __global__ updatepsi(float2* f, float2* ff, float2* ftmp0, float2* ftmp1,
	float2* fff, float rho, float gamma, float maxint, int Ntheta, int Nz,int N)
{
	int tx = blockDim.x * blockIdx.x + threadIdx.x;
	int ty = blockDim.y * blockIdx.y + threadIdx.y;
	int tz = blockDim.z * blockIdx.z + threadIdx.z;

	if (tx>=N||ty>=Nz||tz>=Ntheta) return;

	int ind = tx+ty*N+tz*N*Nz;
	f[ind].x = (1-rho*gamma)*f[ind].x+rho*gamma*(ff[ind].x-fff[ind].x/rho) +
				gamma/2*(ftmp0[ind].x-ftmp1[ind].x)/maxint;
	f[ind].y = (1-rho*gamma)*f[ind].y+rho*gamma*(ff[ind].y-fff[ind].y/rho) +
				gamma/2*(ftmp0[ind].y-ftmp1[ind].y)/maxint;
}

void __global__ takeshifts(float2* shiftx,float2* shifty,float* scanx,float* scany,int Ntheta, int Nscan)
{
	int tx = blockDim.x * blockIdx.x + threadIdx.x;
	int ty = blockDim.y * blockIdx.y + threadIdx.y;

	if (tx>=Nscan||ty>=Ntheta) return;
	int ind = tx+ty*Nscan;
	shiftx[ind].x = cosf(2*PI*(scanx[ind] - roundf(scanx[ind])));
	shiftx[ind].y = sinf(2*PI*(scanx[ind] - roundf(scanx[ind])));
	shifty[ind].x = cosf(2*PI*(scany[ind] - roundf(scany[ind])));
	shifty[ind].y = sinf(2*PI*(scany[ind] - roundf(scany[ind])));
}

void __global__ shifts(float2* f, float2* shiftx,float2* shifty,int Ntheta, int Nscan, int detxdety)
{
	int tx = blockDim.x * blockIdx.x + threadIdx.x;
	int ty = blockDim.y * blockIdx.y + threadIdx.y;
	int tz = blockDim.z * blockIdx.z + threadIdx.z;

	if (tx>=detxdety||ty>=Nscan||tz>=Ntheta) return;
	int ind = tx+ty*detxdety+tz*detxdety*Nscan;
	int inds = ty+tz*Nscan;
	float2 f0 = f[ind];
	float2 shiftx0 = shiftx[inds];
	float2 shifty0 = shifty[inds];
	f[ind].x = f0.x*shiftx0.x-f0.y*shiftx0.y;
	f[ind].y = f0.y*shiftx0.x+f0.x*shiftx0.y;
	f0 = f[ind];
	f[ind].x = f0.x*shifty0.x-f0.y*shifty0.y;
	f[ind].y = f0.y*shifty0.x+f0.x*shifty0.y;
}

void __global__ shiftsa(float2* f, float2* shiftx,float2* shifty,int Ntheta, int Nscan, int detxdety)
{
	int tx = blockDim.x * blockIdx.x + threadIdx.x;
	int ty = blockDim.y * blockIdx.y + threadIdx.y;
	int tz = blockDim.z * blockIdx.z + threadIdx.z;

	if (tx>=detxdety||ty>=Nscan||tz>=Ntheta) return;
	int ind = tx+ty*detxdety+tz*detxdety*Nscan;
	int inds = ty+tz*Nscan;
	float2 f0 = f[ind];
	float2 shiftx0 = shiftx[inds];
	float2 shifty0 = shifty[inds];
	f[ind].x = f0.x*shiftx0.x+f0.y*shiftx0.y;
	f[ind].y = f0.y*shiftx0.x-f0.x*shiftx0.y;
	f0 = f[ind];
	f[ind].x = f0.x*shifty0.x+f0.y*shifty0.y;
	f[ind].y = f0.y*shifty0.x-f0.x*shifty0.y;
}
