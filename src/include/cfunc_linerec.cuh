#ifndef cfunc_linerec_CUH
#define cfunc_linerec_CUH

class cfunc_linerec {
  bool is_free = false;

  float **theta;

public:
  size_t n;      // width of square slices
  size_t ntheta; // number of angles
  size_t pnz;    // number of slices
  float center;  // location of the rotation center
  size_t ngpus;
  cfunc_linerec(size_t ntheta, size_t pnz, size_t n, float center, size_t theta_, size_t ngpus_);
  ~cfunc_linerec();
  void fwd(size_t g, size_t f, size_t igpu);
  void adj(size_t f, size_t g, size_t igpu);
  void free();
};

#endif
