/*interface*/
%module cfunc_fourierrec

%{
#define SWIG_FILE_WITH_INIT
#include "cfunc_fourierrec.cuh"
%}

class cfunc_fourierrec
{
public:
  %immutable;
  size_t n;
  size_t ntheta;
  size_t pnz;
  float center;
  size_t ngpus;

  %mutable;
  cfunc_fourierrec(size_t ntheta, size_t pnz, size_t n, float center, size_t theta_, size_t ngpus);
  ~cfunc_fourierrec();
  void fwd(size_t g, size_t f, size_t igpu);
  void adj(size_t f, size_t g, size_t igpu);
  void free();
};
