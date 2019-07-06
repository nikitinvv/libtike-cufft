#include <cufft.h>

class radonusfft
{
  size_t N;
  size_t Ntheta;
  size_t Nz;
  size_t M;
  float mu;

  float2 *f;
  float2 *g;
  float2 *ff;
  float2 *gg;
  float2 *f0;
  float2 *g0;
  float *theta;

  float *x;
  float *y;

  float2 *fde;
  float2 *fdee;


  cufftHandle plan2dfwd;
  cufftHandle plan2dadj;

  cufftHandle plan1d;

public:
  radonusfft(size_t Ntheta, size_t Nz, size_t N);
  ~radonusfft();
  void fwd(size_t g, size_t f);
  void adj(size_t f, size_t g);
  void setobj(size_t theta);
};
