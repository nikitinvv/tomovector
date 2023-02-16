/*interface*/
%module cfunc_linerec

%{
#define SWIG_FILE_WITH_INIT
#include "cfunc_linerec.cuh"
%}

class cfunc_linerec
{
public:
  %immutable;
  size_t n;
  size_t ntheta;
  size_t pnz;
  float center;
  size_t ngpus;

  %mutable;
  cfunc_linerec(size_t ntheta, size_t pnz, size_t n, float center, size_t theta_, size_t ngpus);
  ~cfunc_linerec();
  void fwd(size_t g, size_t f, size_t igpu);
  void adj(size_t f, size_t g, size_t igpu);
  void free();
};
