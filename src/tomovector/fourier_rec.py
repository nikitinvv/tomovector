from tomovector.cfunc_fourierrec import cfunc_fourierrec
import cupy as cp


class FourierRec():
    """Fourier-based method"""

    def __init__(self, n, ntheta, pnz, theta, center, ngpus):
        self.theta = theta  # keep theta in memory
        self.pnz = pnz
        self.n = n
        self.ntheta = ntheta
        self.cl = cfunc_fourierrec(ntheta, pnz//2, n, center, theta.ctypes.data, ngpus)

    def free(self):
        self.cl.free()
    
    def fwd(self, u, gpu=0):
        """Radon transform (R)"""
        res = cp.zeros([self.pnz//2,  self.ntheta, 2*self.n], dtype='float32')
        u = cp.ascontiguousarray(cp.concatenate(
            (u[:self.pnz//2, :, :, cp.newaxis], u[self.pnz//2:, :, :, cp.newaxis]), axis=3).reshape(u.shape))
        self.cl.fwd(res.data.ptr, u.data.ptr, gpu)
        res = cp.concatenate((res[..., ::2], res[..., 1::2]),axis=0)        
        return res
    
    def adj(self, data, gpu=0):
        """Adjoint Radon transform (R^*)"""
        data = cp.ascontiguousarray(cp.concatenate(
            (data[:self.pnz//2, :, :, cp.newaxis], data[self.pnz//2:, :, :, cp.newaxis]), axis=3).reshape(data.shape))
        res = cp.zeros([self.pnz//2, self.n, 2*self.n], dtype='float32')
        self.cl.adj(res.data.ptr, data.data.ptr, gpu)
        res = cp.concatenate((res[..., ::2], res[..., 1::2]),axis=0)
        return res