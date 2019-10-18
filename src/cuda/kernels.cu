// This module defines CUDA kernels for ptychography.

// The main function is compute_indices which computes calculates the index that
// each thread needs to do its work on the f, g, and prb arrays for the various
// kernels defined here.

#define PI 3.1415926535

// Compute the array index for a thread in the f, g, and prb matricies.
void __device__ compute_indices(
  size_t* const f_index, size_t* const g_index, size_t* const prb_index,
  const float * const scanx, const float * const scany,
  const int Ntheta, const int Nz, const int N, const int Nscan, const int Nprb,
  const int Ndetx, const int Ndety)
{
  const int tx = blockDim.x * blockIdx.x + threadIdx.x;
  const int ty = blockDim.y * blockIdx.y + threadIdx.y;
  const int tz = blockDim.z * blockIdx.z + threadIdx.z;
  if (tx >= Nprb * Nprb || ty >= Nscan || tz >= Ntheta) return;

  // coordinates in the probe array
  const int ix = tx % Nprb;
  const int iy = tx / Nprb;

  if (f_index != NULL)
  {
    // closest integers for scan positions
    const int stx = roundf(scanx[ty + tz * Nscan]);
    const int sty = roundf(scany[ty + tz * Nscan]);
    // skip scans where the probe position is negative (undefined)
    if (stx < 0 || sty < 0) return;

    // coordinates in the f array
    *f_index = (
      + (stx + ix)
      + (sty + iy) * N
      + tz * Nz * N
    );
  }

  if (g_index != NULL)
  {
    // coordinates in the g array
    *g_index = (
      // shift in the object array to the starting point of probe multiplication
      + (Ndety - Nprb) / 2 * Ndetx
      + (Ndetx - Nprb) / 2
      // shift in the object array multilication for this thread
      + ix
      + iy * Ndetx
      + ty * Ndetx * Ndety
      + tz * Ndetx * Ndety * Nscan
    );
  }

  if (prb_index != NULL)
  {
    // coordinates in the probe array
    *prb_index = (
      + ix
      + iy * Nprb
      + tz * Nprb * Nprb
    );
  }
}

// Multiply the probe array (prb) by the patches (g) extracted from the object.
void __global__ mulprobe(
  float2 *f, float2 *g, float2 *prb,
  float *scanx, float *scany,
  int Ntheta, int Nz, int N, int Nscan, int Nprb, int Ndetx, int Ndety)
{
  size_t g_index = 0, prb_index = 0;
  compute_indices(
    NULL, &g_index, &prb_index,
    scanx, scany, Ntheta, Nz, N, Nscan, Nprb, Ndetx, Ndety);
  float2 g0 = g[g_index];
  float2 prb0 = prb[prb_index];
  // multiplication in complex variables
  const float c = 1 / sqrtf(Ndetx * Ndety); // fft constant
  g[g_index].x = c * (prb0.x * g0.x - prb0.y * g0.y);
  g[g_index].y = c * (prb0.x * g0.y + prb0.y * g0.x);
}

// Multiply the object patches (g) by the complex conjugate of the probe (prb).
void __global__ mulaprobe(
  float2 *f, float2 *g, float2 *prb,
  float *scanx, float *scany,
  int Ntheta, int Nz, int N, int Nscan, int Nprb, int Ndetx, int Ndety)
{
  size_t g_index = 0, prb_index = 0;
  compute_indices(
    NULL, &g_index, &prb_index,
    scanx, scany, Ntheta, Nz, N, Nscan, Nprb, Ndetx, Ndety);
  float2 g0 = g[g_index];
  float2 prb0 = prb[prb_index];
  // multiplication in complex variables
  const float c = 1 / sqrtf(Ndetx * Ndety); // fft constant
  g[g_index].x = c * (prb0.x * g0.x + prb0.y * g0.y);
  g[g_index].y = c * (prb0.x * g0.y - prb0.y * g0.x);
}

// Multiply the object patches (g) by the complex conjugate of the object (f).
void __global__ mulaobj(
  float2 *f, float2 *g, float2 *prb,
  float *scanx, float *scany,
  int Ntheta, int Nz, int N, int Nscan, int Nprb, int Ndetx, int Ndety)
{
  size_t f_index = 0, g_index = 0;
  compute_indices(
    &f_index, &g_index, NULL,
    scanx, scany, Ntheta, Nz, N, Nscan, Nprb, Ndetx, Ndety);
  float2 f0 = f[f_index];
  float2 g0 = g[g_index];
  // multiplication in complex variables
  const float c = 1 / sqrtf(Ndetx * Ndety); // fft constant
  g[g_index].x = c * (f0.x * g0.x + f0.y * g0.y);
  g[g_index].y = c * (f0.x * g0.y - f0.y * g0.x);
}

// Take patches of the object array (f) where the probe illuminates and put them
// into the far-field array (g).
void __global__ takepart(
  float2 *f, float2 *g, float2 *prb,
  float *scanx, float *scany,
  int Ntheta, int Nz, int N, int Nscan, int Nprb, int Ndetx, int Ndety)
{
  size_t f_index = 0, g_index = 0;
  compute_indices(
    &f_index, &g_index, NULL,
    scanx, scany, Ntheta, Nz, N, Nscan, Nprb, Ndetx, Ndety);
  g[g_index].x = f[f_index].x;
  g[g_index].y = f[f_index].y;
}

// Add the object patches (g) to the object array (f).
void __global__ setpartobj(
  float2 *f, float2 *g, float2 *prb,
  float *scanx, float *scany,
  int Ntheta, int Nz, int N, int Nscan, int Nprb, int Ndetx, int Ndety)
{
  size_t f_index = 0, g_index = 0;
  compute_indices(
    &f_index, &g_index, NULL,
    scanx, scany, Ntheta, Nz, N, Nscan, Nprb, Ndetx, Ndety);
  atomicAdd(&f[f_index].x, g[g_index].x);
  atomicAdd(&f[f_index].y, g[g_index].y);
}

// Add the object patches (g) to the probe array (prb).
void __global__ setpartprobe(
  float2 *f, float2 *g, float2 *prb,
  float *scanx, float *scany,
	int Ntheta, int Nz, int N, int Nscan, int Nprb, int Ndetx, int Ndety)
{
  size_t g_index = 0, prb_index = 0;
  compute_indices(
    NULL, &g_index, &prb_index,
    scanx, scany, Ntheta, Nz, N, Nscan, Nprb, Ndetx, Ndety);
  atomicAdd(&prb[prb_index].x, g[g_index].x);
  atomicAdd(&prb[prb_index].y, g[g_index].y);
}
