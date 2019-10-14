// This module defines interpolation related CUDA kernels for ptychography.

// Compute exp(1j * dir * dx), exp(1j * dir * dy) where dx, dy are in (-1, 1)
// and correspond to shifts to nearest integer. dir is the shift direction: 1
// for forward and -1 for backward.
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

// Perform shifts in the frequency domain by multiplication with exp(1j * dx),
// exp(1j * dy).
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
