// Ptychography kernels on GPU

#define PI 3.1415926535

// probe multiplication of the object array part
void __global__ mulprobe(float2 *g, float2 *f, float2 *prb,
                         float *scanx, float *scany,
                         int Ntheta, int Nz, int N, int Nscan, int Nprb,
                         int Ndetx, int Ndety)
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
void __global__ mulaprobe(float2 *f, float2 *g, float2 *prb, float *scanx, float *scany,
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
void __global__ mulaobj(float2 *prb, float2 *g, float2 *f, float *scanx, float *scany,
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
void __global__ takepart(float2 *g, float2 *f, float2 *prb, float *scanx, float *scany,
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
void __global__ setpartobj(float2 *f, float2 *g, float2 *prb, float *scanx, float *scany,
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
void __global__ setpartprobe(float2 *prb, float2 *g, float2 *f, float *scanx, float *scany,
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

// Compute exp(1j * dir * dx), exp(1j * dir * dy) where dx, dy are in (-1, 1)
// and correspond to shifts to nearest integer.
// dir is the shift direction: 1 for forward and -1 for backward.
void __global__ takeshifts(float2 *shiftx, float2 *shifty, float *scanx,
                           float *scany, int dir, int Ntheta, int Nscan)
{
  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int ty = blockDim.y * blockIdx.y + threadIdx.y;
  if (tx >= Nscan || ty >= Ntheta)
    return;

  int ind = tx + ty * Nscan;
  float intpart;  // modf requires a place to save the integer part

  // compute exp(1j * dx), exp(1j * dy) in terms of cos and sin
  shiftx[ind].x = cosf(2 * PI * dir * modff(scanx[ind], &intpart));
  shiftx[ind].y = sinf(2 * PI * dir * modff(scanx[ind], &intpart));
  shifty[ind].x = cosf(2 * PI * dir * modff(scany[ind], &intpart));
  shifty[ind].y = sinf(2 * PI * dir * modff(scany[ind], &intpart));
}

// Perform shifts in the frequency domain by multiplication
// with exp(1j * dx), exp(1j * dy)
void __global__ shifts(float2 *f, float2 *shiftx, float2 *shifty,
                       int Ntheta, int Nscan, int NdetxNdety, int NprbNprb)
{
  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int ty = blockDim.y * blockIdx.y + threadIdx.y;
  int tz = blockDim.z * blockIdx.z + threadIdx.z;
  if (tx >= NdetxNdety || ty >= Nscan || tz >= Ntheta)
    return;

  int ind = tx + ty * NdetxNdety + tz * NdetxNdety * Nscan;
  int inds = ty + tz * Nscan;

  // multiply f with exp(1j * dx)
  float2 f0 = f[ind];
  f[ind].x = f0.x * shiftx[inds].x - f0.y * shiftx[inds].y;
  f[ind].y = f0.y * shiftx[inds].x + f0.x * shiftx[inds].y;
  // multiply f with exp(1j * dy)
  f0 = f[ind];
  float c = 1 / (float)(NprbNprb); // fft constant for shifts
  f[ind].x = c * (f0.x * shifty[inds].x - f0.y * shifty[inds].y);
  f[ind].y = c * (f0.y * shifty[inds].x + f0.x * shifty[inds].y);
}
