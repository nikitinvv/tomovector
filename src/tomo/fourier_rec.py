from tomo.cfunc_fourierrec import cfunc_fourierrec
import cupy as cp


class FourierRec():
    """Fourier-based method"""

    def __init__(self, n, ntheta, nz, theta, center):
        self.theta = theta  # keep theta in memory
        self.nz = nz
        self.n = n
        self.ntheta = ntheta
        self.ne = 3*n//2
        self.cl = cfunc_fourierrec(ntheta, nz//2, n, center, theta.data.ptr, 1)
    
    def __enter__(self):
        """Return self at start of a with-block."""
        return self

    def __exit__(self, type, value, traceback):
        """Free GPU memory due at interruptions or with-block exit."""
        self.cl.free()
        
    def fwd(self, u, gpu=0):
        """Radon transform (R)"""
        res = cp.zeros([self.nz//2,  self.ntheta, 2*self.n], dtype='float32')
        u = cp.ascontiguousarray(cp.concatenate(
            (u[:self.nz//2, :, :, cp.newaxis], u[self.nz//2:, :, :, cp.newaxis]), axis=3).reshape(u.shape))
        self.cl.fwd(res.data.ptr, u.data.ptr, gpu)
        res = cp.concatenate((res[..., ::2], res[..., 1::2]),axis=0)        
        return res
    
    def adj(self, data, gpu=0):
        """Adjoint Radon transform (R^*)"""
        data = cp.ascontiguousarray(cp.concatenate(
            (data[:self.nz//2, :, :, cp.newaxis], data[self.nz//2:, :, :, cp.newaxis]), axis=3).reshape(data.shape))
        res = cp.zeros([self.nz//2, self.n, 2*self.n], dtype='float32')
        self.cl.adj(res.data.ptr, data.data.ptr, gpu)
        res = cp.concatenate((res[..., ::2], res[..., 1::2]),axis=0)
        return res

    def fbp_filter(self, data, fbp_filter='parzen'):
        """FBP filtering of projections"""
        
        t = cp.fft.rfftfreq(self.ne).astype('float32')
        if fbp_filter == 'parzen':
            w = t * (1 - t * 2)**3
        elif fbp_filter == 'shepp':
            w = t * cp.sinc(t)
        elif fbp_filter == 'ramp':
            w = t

        tmp = cp.pad(data, ((0, 0), (0, 0), (self.ne//2-self.n//2, self.ne//2-self.n//2)), mode='edge')        
        tmp = cp.fft.irfft(w*cp.fft.rfft(tmp, axis=2), axis=2)        
        data = tmp[:, :, self.ne//2-self.n//2:self.ne//2+self.n//2]/self.n*2

        return data