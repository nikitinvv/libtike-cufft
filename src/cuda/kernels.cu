// This module defines CUDA kernels for ptychography.

// The main function is muloperator which computes multiplication by the probe
// function or object function so as their adjoints. The operation is performed
// with respect to indices for threads in the f, g, and prb matricies. Skip
// computations if probe position is negative.

void __global__ muloperator(float2 *f, float2 *g, float2 *prb,
  const float2 * const scan,
  const int Ntheta, const int Nz, const int N, const int Nscan, const int Nprb,
  const int ndet, int flg)
{
  const int tx = blockDim.x * blockIdx.x + threadIdx.x;
  const int ty = blockDim.y * blockIdx.y + threadIdx.y;
  const int tz = blockDim.z * blockIdx.z + threadIdx.z;
  if (tx >= Nprb * Nprb || ty >= Nscan || tz >= Ntheta) return;

  // coordinates in the probe array
  const int ix = tx % Nprb;
  const int iy = tx / Nprb;

  float sx;  // modf requires a place to save the integer part
  float sy;
  float sxf ;
  float syf;
  
  sxf = modff(scan[ty + tz * Nscan].y, &sx);
  syf = modff(scan[ty + tz * Nscan].x, &sy);

  sxf=0;
  syf=0; // temporarily for TESTING

  //sx = (int)(scan[ty + tz * Nscan].y);
  //sy = (int)(scan[ty + tz * Nscan].x);
  //sxf = (scan[ty + tz * Nscan].y)-sx;
  //syf = (scan[ty + tz * Nscan].x)-sy;
  
  // skip scans where the probe position is negative (undefined)
  if (sx < 0 || sy < 0) return;

  // coordinates in the f array
  int f_index = (
      + (sx + ix)
      + (sy + iy) * N
      + tz * Nz * N
    );
  // coordinates in the g array
  int g_index = (
      // shift probe multiplication to min corner of nearplane array so that
      // FFTS are correct when probe and farplane sizes mismatch
      + (ndet - Nprb) / 2 * (ndet + 1)
      // shift in the nearplane array multiplication for this thread
      + ix
      + iy * ndet
      + ty * ndet * ndet
      + tz * ndet * ndet * Nscan
    );
  // coordinates in the probe array
  int prb_index = (
      + ix
      + iy * Nprb
      + tz * Nprb * Nprb
    );

  const float c = 1.0 / static_cast<float>(ndet); // fft constant
  float2 tmp; //tmp variable
  
  // Linear interpolation
  if(flg==0) //adjoint
  {
    tmp.x = c * (prb[prb_index].x * g[g_index].x + prb[prb_index].y * g[g_index].y);
    tmp.y = c * (prb[prb_index].x * g[g_index].y - prb[prb_index].y * g[g_index].x);
    atomicAdd(&f[f_index].x,     tmp.x*(1-sxf)*(1-syf));
    atomicAdd(&f[f_index].y,     tmp.y*(1-sxf)*(1-syf));
    atomicAdd(&f[f_index+1].x,   tmp.x*(sxf  )*(1-syf));
    atomicAdd(&f[f_index+1].y,   tmp.y*(sxf  )*(1-syf));
    atomicAdd(&f[f_index+N].x,   tmp.x*(1-sxf)*(syf  ));
    atomicAdd(&f[f_index+N].y,   tmp.y*(1-sxf)*(syf  ));
    atomicAdd(&f[f_index+1+N].x, tmp.x*(sxf  )*(syf  ));
    atomicAdd(&f[f_index+1+N].y, tmp.y*(sxf  )*(syf  ));
  }
  else if(flg==1) //adjoint probe
  {
    tmp.x = f[f_index].x   *(1-sxf)*(1-syf)+
           f[f_index+1].x  *(sxf  )*(1-syf)+
           f[f_index+N].x  *(1-sxf)*(syf  )+
           f[f_index+1+N].x*(sxf  )*(syf  );
    tmp.y = f[f_index].y  *(1-sxf)*(1-syf)+
           f[f_index+1].y  *(sxf  )*(1-syf)+
           f[f_index+N].y  *(1-sxf)*(syf  )+
           f[f_index+1+N].y*(sxf  )*(syf  );
           atomicAdd(&prb[prb_index].x, c * (g[g_index].x * tmp.x + g[g_index].y * tmp.y));
           atomicAdd(&prb[prb_index].y, c * (g[g_index].y * tmp.x - g[g_index].x * tmp.y));
  }
  else if (flg==2) //forward
  {
    tmp.x = f[f_index].x   *(1-sxf)*(1-syf)+
           f[f_index+1].x  *(sxf  )*(1-syf)+
           f[f_index+N].x  *(1-sxf)*(syf  )+
           f[f_index+1+N].x*(sxf  )*(syf  );
    tmp.y = f[f_index].y  *(1-sxf)*(1-syf)+
           f[f_index+1].y  *(sxf  )*(1-syf)+
           f[f_index+N].y  *(1-sxf)*(syf  )+
           f[f_index+1+N].y*(sxf  )*(syf  );
    g[g_index].x = c * (prb[prb_index].x * tmp.x - prb[prb_index].y * tmp.y);
    g[g_index].y = c * (prb[prb_index].x * tmp.y + prb[prb_index].y * tmp.x);
  }
}
