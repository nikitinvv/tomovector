void __global__ fwd_ker(float *data, float *f, float *theta, float center, float c, int n, int ntheta, int pnz)
{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    int tz = blockDim.z * blockIdx.z + threadIdx.z;
    if (tx >= n || ty >= ntheta || tz >= pnz)
        return;
    float x,y;
    int xr,yr;
    
    float data0 = 0;
    float theta0 = theta[ty];
    float ctheta = __cosf(theta0);
    float stheta = __sinf(theta0);
        
    for (int t = 0; t<n; t++)
    {
        x = ctheta*(tx-n/2)+stheta*(t-n/2) + center;
        y = -stheta*(tx-n/2)+ctheta*(t-n/2) + center;
        xr = (int)(x-1e-5);
        yr = (int)(y-1e-5);
        // linear interp            
        if ((xr >= 0) & (xr < n - 1) & (yr >= 0) & (yr < n-1))
        {
            x = x-xr;
            y = y-yr;
            data0 += f[xr+0+(yr+0)*n+tz*n*n]*(1-x)*(1-y)+
                    f[xr+1+(yr+0)*n+tz*n*n]*(0+x)*(1-y)+
                    f[xr+0+(yr+1)*n+tz*n*n]*(1-x)*(0+y)+
                    f[xr+1+(yr+1)*n+tz*n*n]*(0+x)*(0+y);
        }
    }
    data[tx + ty * n + tz * n * ntheta] += data0*c; 
}    

void __global__ adj_ker(float *f, float *data, float *theta, float center, float c, int n, int ntheta, int pnz)
{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    int tz = blockDim.z * blockIdx.z + threadIdx.z;
    if (tx >= n || ty >= n || tz >= pnz)
        return;
    float u = 0;
    int ur = 0;
    
    float f0 = 0;
    float theta0 = 0;
    
    for (int t = 0; t<ntheta; t++)
    {
        theta0 = theta[t];            
        float ctheta = __cosf(theta0);
        float stheta = __sinf(theta0);
        u = ctheta*(tx-n/2)+stheta*(ty-n/2)+center;        
        ur = (int)(u-1e-5);
        
        // linear interp            
        if ((ur >= 0) & (ur < n - 1))
        {
            u = u-ur;
            f0 +=   data[ur+0+t*n+tz*n*ntheta]*(1-u)+
                    data[ur+1+t*n+tz*n*ntheta]*(0+u);                    
        }
    }
    f[tx + (n-1-ty) * n + tz * n * n] += f0*c; 
}    