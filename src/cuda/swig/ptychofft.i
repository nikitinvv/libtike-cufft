/*interface*/
%module ptychofft

%{
#define SWIG_FILE_WITH_INIT
#include "ptychofft.cuh"
%}

class ptychofft {
public:
  %immutable;
  size_t ptheta; // number of projections
  size_t nz;     // object vertical size
  size_t n;      // object horizontal size
  size_t nscan;  // number of scan positions for 1 projection
  size_t ndet;   // detector y size
  size_t nprb;   // probe size in 1 dimension

  %mutable;
  ptychofft(size_t ptheta, size_t nz, size_t n, size_t nscan,
            size_t detector_shape, size_t probe_shape);
  ~ptychofft();
  void fwd(size_t g_, size_t f_, size_t scan_, size_t prb_);
  void adj(size_t f_, size_t g_, size_t scan_, size_t prb_, int flg);
  void free();
};
