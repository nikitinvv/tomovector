#include "cfunc_linerec.cuh"
#include "kernels_linerec.cu"
#include <stdio.h>

cfunc_linerec::cfunc_linerec(size_t ntheta, size_t pnz, size_t n, float center, size_t theta_, size_t ngpus)
  : ntheta(ntheta), pnz(pnz), n(n), center(center), ngpus(ngpus) 
  {
    theta = new float*[ngpus];
    for (int igpu=0;igpu<ngpus;igpu++)
    {
      cudaSetDevice(igpu);
      cudaMalloc((void **)&theta[igpu], ntheta * sizeof(float));
      cudaMemcpy(theta[igpu], (float *)theta_, ntheta * sizeof(float), cudaMemcpyDefault);
    }
    //back to 0
    cudaSetDevice(0);
  }

// destructor, memory deallocation
cfunc_linerec::~cfunc_linerec() { free(); }


void cfunc_linerec::free() {
  if (!is_free)  
    is_free = true;     
}

void cfunc_linerec::fwd(size_t g_, size_t f_, size_t igpu) 
{
    cudaSetDevice(igpu);
    float* g = (float *)g_;    
    float* f = (float *)f_;
    // set thread block, grid sizes will be computed before cuda kernel execution
    dim3 dimBlock(32,32,1);    
    dim3 GS3d0;  
    GS3d0 = dim3(ceil(n / 32.0), ceil(ntheta / 32.0), pnz);
    fwd_ker <<<GS3d0, dimBlock>>> (g, f, theta[igpu], center, 1, n, ntheta, pnz);
}   

void cfunc_linerec::adj(size_t f_, size_t g_, size_t igpu) 
{
    cudaSetDevice(igpu);
    float* g = (float *)g_;    
    float* f = (float *)f_;
    dim3 dimBlock(32,32,1);    
    dim3 GS3d0;  
    GS3d0 = dim3(ceil(n / 32.0), ceil(n / 32.0), pnz);
    adj_ker <<<GS3d0, dimBlock>>> (f, g, theta[igpu], center, 1, n, ntheta, pnz);
}
