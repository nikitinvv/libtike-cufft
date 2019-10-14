// Ptychography kernels on GPU

#define PI 3.1415926535

// probe multiplication of the object array part
void __global__ mulprobe(
  float2 *f, float2 *g, float2 *prb,
  float *scanx, float *scany,
  int Ntheta, int Nz, int N, int Nscan, int Nprb, int Ndetx, int Ndety)
{
  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int ty = blockDim.y * blockIdx.y + threadIdx.y;
  int tz = blockDim.z * blockIdx.z + threadIdx.z;
  if (tx >= Nprb * Nprb || ty >= Nscan || tz >= Ntheta)
    return;

  // coordinates in the probe array
  int ix = tx % Nprb;
  int iy = tx / Nprb;
  // probe voxel for multiplication
  float2 prb0 = prb[
    + ix
    + iy * Nprb
    + tz * Nprb * Nprb
  ];

  // coordinates in the object array
  size_t shift = (
    // shift in the object array to the starting point of probe multiplication
    + (Ndety - Nprb) / 2 * Ndetx
    + (Ndetx - Nprb) / 2
    // shift in the object array multilication for this thread
    + ix
    + iy * Ndetx
    + ty * Ndetx * Ndety
    + tz * Ndetx * Ndety * Nscan
  );
  // object voxel for multiplication
  float2 g0 = g[shift];

  // multiplication in complex variables
  float c = 1 / sqrtf(Ndetx * Ndety); // fft constant
  g[shift].x = c * (prb0.x * g0.x - prb0.y * g0.y);
  g[shift].y = c * (prb0.x * g0.y + prb0.y * g0.x);
}

// adjoint probe multiplication of the object array part
void __global__ mulaprobe(
  float2 *f, float2 *g, float2 *prb,
  float *scanx, float *scany,
  int Ntheta, int Nz, int N, int Nscan, int Nprb, int Ndetx, int Ndety)
{
  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int ty = blockDim.y * blockIdx.y + threadIdx.y;
  int tz = blockDim.z * blockIdx.z + threadIdx.z;
  if (tx >= Nprb * Nprb || ty >= Nscan || tz >= Ntheta)
    return;

  // closest integers for scan positions
  int stx = roundf(scanx[ty + tz * Nscan]);
  int sty = roundf(scany[ty + tz * Nscan]);
  // skip scans where the probe position is negative (undefined)
  if (stx < 0 || sty < 0)
    return;

  // coordinates in the probe array
  int ix = tx % Nprb;
  int iy = tx / Nprb;
  // probe voxel for multiplication
  float2 prb0 = prb[
    + ix
    + iy * Nprb
    + tz * Nprb * Nprb
  ];

  // coordinates in the object array
  size_t shift = (
    // shift in the object array to the starting point of probe multiplication
    + (Ndety - Nprb) / 2 * Ndetx
    + (Ndetx - Nprb) / 2
    // shift in the object array multilication for this thread
    + ix
    + iy * Ndetx
    + ty * Ndetx * Ndety
    + tz * Ndetx * Ndety * Nscan
  );
  // data voxel for multiplication
  float2 g0 = g[shift];

  // multiplication in complex variables
  float c = 1 / sqrtf(Ndetx * Ndety); // fft constant
  g[shift].x = c * (prb0.x * g0.x + prb0.y * g0.y);
  g[shift].y = c * (prb0.x * g0.y - prb0.y * g0.x);
}

// adjoint object part multiplication of the probe array
void __global__ mulaobj(
  float2 *f, float2 *g, float2 *prb,
  float *scanx, float *scany,
  int Ntheta, int Nz, int N, int Nscan, int Nprb, int Ndetx, int Ndety)
{
  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int ty = blockDim.y * blockIdx.y + threadIdx.y;
  int tz = blockDim.z * blockIdx.z + threadIdx.z;
  if (tx >= Nprb * Nprb || ty >= Nscan || tz >= Ntheta)
    return;

  // closest integers for scan positions
  int stx = roundf(scanx[ty + tz * Nscan]);
  int sty = roundf(scany[ty + tz * Nscan]);
  // skip scans where the probe position is negative (undefined)
  if (stx < 0 || sty < 0)
    return;

  // coordinates in the probe array
  int ix = tx % Nprb;
  int iy = tx / Nprb;
  // object voxel for multiplication
  float2 f0 = f[
    + (stx + ix)
    + (sty + iy) * N
    + tz * Nz * N
  ];

  // coordinates in the object array
  size_t shift = (
    // shift in the object array to the starting point of probe multiplication
    + (Ndety - Nprb) / 2 * Ndetx
    + (Ndetx - Nprb) / 2
    // shift in the object array multilication for this thread
    + ix
    + iy * Ndetx
    + ty * Ndetx * Ndety
    + tz * Ndetx * Ndety * Nscan
  );
  // data voxel for multiplication
  float2 g0 = g[shift];

  // multiplication in complex variables
  float c = 1 / sqrtf(Ndetx * Ndety); // fft constant
  g[shift].x = c * (f0.x * g0.x + f0.y * g0.y);
  g[shift].y = c * (f0.x * g0.y - f0.y * g0.x);
}

// take part of the object array
void __global__ takepart(
  float2 *f, float2 *g, float2 *prb,
  float *scanx, float *scany,
  int Ntheta, int Nz, int N, int Nscan, int Nprb, int Ndetx, int Ndety)
{
  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int ty = blockDim.y * blockIdx.y + threadIdx.y;
  int tz = blockDim.z * blockIdx.z + threadIdx.z;
  if (tx >= Nprb * Nprb || ty >= Nscan || tz >= Ntheta)
  return;

  // coordinates in the probe array
  int ix = tx % Nprb;
  int iy = tx / Nprb;
  // closest integers for scan positions
  int stx = roundf(scanx[ty + tz * Nscan]);
  int sty = roundf(scany[ty + tz * Nscan]);
  // skip scans where the probe position is negative (undefined)
  if (stx < 0 || sty < 0)
    return;

  // take the part
  float2 f0 = f[(stx + ix) + (sty + iy) * N + tz * Nz * N];

  // coordinates in the object array
  size_t shift = (
    // shift in the object array to the starting point of probe multiplication
    + (Ndety - Nprb) / 2 * Ndetx
    + (Ndetx - Nprb) / 2
    // shift in the object array multilication for this thread
    + ix
    + iy * Ndetx
    + ty * Ndetx * Ndety
    + tz * Ndetx * Ndety * Nscan
  );
  g[shift].x = f0.x;
  g[shift].y = f0.y;
}

// simultaneous writing to the object array
void __global__ setpartobj(
  float2 *f, float2 *g, float2 *prb,
  float *scanx, float *scany,
  int Ntheta, int Nz, int N, int Nscan, int Nprb, int Ndetx, int Ndety)
{
  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int ty = blockDim.y * blockIdx.y + threadIdx.y;
  int tz = blockDim.z * blockIdx.z + threadIdx.z;

  if (tx >= Nprb * Nprb || ty >= Nscan || tz >= Ntheta)
    return;

  // coordinates in the probe array
  int ix = tx % Nprb;
  int iy = tx / Nprb;
  // closest integers for scan positions
  int stx = roundf(scanx[ty + tz * Nscan]);
  int sty = roundf(scany[ty + tz * Nscan]);
  // skip scans where the probe position is negative (undefined)
  if (stx < 0 || sty < 0)
    return;

  // coordinates in the object array
  size_t shift = (
    // shift in the object array to the starting point of probe multiplication
    + (Ndety - Nprb) / 2 * Ndetx
    + (Ndetx - Nprb) / 2
    // shift in the object array multilication for this thread
    + ix
    + iy * Ndetx
    + ty * Ndetx * Ndety
    + tz * Ndetx * Ndety * Nscan
  );
  float2 g0 = g[shift]; // data voxel for multiplication

  // multilication in complex variables with simultaneous writing to the object array
  atomicAdd(&f[(stx + ix) + (sty + iy) * N + tz * Nz * N].x, g0.x);
  atomicAdd(&f[(stx + ix) + (sty + iy) * N + tz * Nz * N].y, g0.y);
}

// simultaneous writing to the probe array
void __global__ setpartprobe(
  float2 *f, float2 *g, float2 *prb,
  float *scanx, float *scany,
	int Ntheta, int Nz, int N, int Nscan, int Nprb, int Ndetx, int Ndety)
{
  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int ty = blockDim.y * blockIdx.y + threadIdx.y;
  int tz = blockDim.z * blockIdx.z + threadIdx.z;
  if (tx >= Nprb * Nprb || ty >= Nscan || tz >= Ntheta)
    return;

  // coordinates in the probe array
  int ix = tx % Nprb;
  int iy = tx / Nprb;
  // closest integers for scan positions
  int stx = roundf(scanx[ty + tz * Nscan]);
  int sty = roundf(scany[ty + tz * Nscan]);
  // skip scans where the probe position is negative (undefined)
  if (stx < 0 || sty < 0)
    return;

  // coordinates in the object array
  size_t shift = (
    // shift in the object array to the starting point of probe multiplication
    + (Ndety - Nprb) / 2 * Ndetx
    + (Ndetx - Nprb) / 2
    // shift in the object array multilication for this thread
    + ix
    + iy * Ndetx
    + ty * Ndetx * Ndety
    + tz * Ndetx * Ndety * Nscan
  );
  float2 g0 = g[shift]; // data voxel for multiplication

  // multilication in complex variables with simultaneous writing to the object array
  atomicAdd(&prb[ix + iy * Nprb + tz * Nprb * Nprb].x, g0.x);
  atomicAdd(&prb[ix + iy * Nprb + tz * Nprb * Nprb].y, g0.y);
}
