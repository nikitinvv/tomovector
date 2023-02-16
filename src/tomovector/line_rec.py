from tomovector.cfunc_linerec import cfunc_linerec
import cupy as cp


class LineRec():
    """Direct line discretization method"""

    def __init__(self, n, ntheta, pnz, theta, center, ngpus):
        self.theta = theta  # keep theta in memory
        self.pnz = pnz
        self.n = n
        self.ntheta = ntheta
        self.cl = cfunc_linerec(ntheta, pnz, n, center, theta.ctypes.data, ngpus)

    def free(self):
        self.cl.free()
        
    def fwd(self, u, gpu=0):
        """Radon transform (R)"""
        res = cp.zeros([self.pnz,  self.ntheta, self.n], dtype='float32')
        u = cp.ascontiguousarray(u)        
        self.cl.fwd(res.data.ptr, u.data.ptr, gpu)        
        return res
    
    def adj(self, data, gpu=0):
        """Adjoint Radon transform (R^*)"""
        res = cp.zeros([self.pnz, self.n, self.n], dtype='float32')
        data = cp.ascontiguousarray(data)                
        self.cl.adj(res.data.ptr, data.data.ptr, gpu)        
        return res