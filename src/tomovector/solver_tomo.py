"""Module for tomography."""

import cupy as cp
import numpy as np
from tomovector.fourier_rec import FourierRec
from tomovector.line_rec import LineRec
import threading
import concurrent.futures as cf
from functools import partial


class SolverTomo():
    """Base class for tomography solvers using the USFFT method on GPU.
    This class is a context manager which provides the basic operators required
    to implement a tomography solver. It also manages memory automatically,
    and provides correct cleanup for interruptions or terminations.
    Attribtues
    ----------
    ntheta : int
        The number of projections.    
    n, nz : int
        The pixel width and height of the projection.
    pnz : int
        The number of slice partitions to process by a GPU simultaneously.
    ngpus : int
        Number of gpus        
    """

    def __init__(self, theta, p, ntheta, nz, n, pnz, center, method, ngpus):
        """Please see help(SolverTomo) for more info."""
        # create class for the tomo transform associated with first gpu
        if method == 'fourierrec':
            self.cl_rec = FourierRec(n, ntheta, pnz, theta, center, ngpus)
        elif method == 'linerec':
            self.cl_rec = LineRec(n, ntheta, pnz, theta, center, ngpus)
        self.p = cp.array(p)
        self.n = n
        self.nz = nz
        self.pnz = pnz
        self.ntheta = ntheta
        self.ngpus = ngpus


    def __enter__(self):
        """Return self at start of a with-block."""
        return self

    def __exit__(self, type, value, traceback):
        """Free GPU memory due at interruptions or with-block exit."""
        self.cl_rec.free()

    def fwd_tomo(self, u, gpu=0):
        """Radon transform (R)"""
        res = cp.zeros([self.pnz,  self.ntheta, self.n], dtype='float32')
        for k in range(3):
            res += self.cl_rec.fwd(u[k], gpu)*self.p[k]                        
        return res

    def adj_tomo(self, data, gpu=0):
        """Adjoint Radon transform (R^*)"""
        res = cp.zeros([3, self.pnz, self.n, self.n], dtype='float32')
        
        for k in range(3):        
            res[k] = self.cl_rec.adj(data*self.p[k], gpu)                        
        return res

    # batched versions of operators
    def fwd_tomo_batch(self, u):
        """Batch of Tomography transform (R)"""
        res = np.zeros([2,self.nz,self.ntheta,self.n], dtype='float32')
        for k in range(self.nz//self.pnz):
            ids = np.arange(k*self.pnz, (k+1)*self.pnz)
            # copy data part to gpu
            u_gpu = cp.array(u[:,ids])
            # Radon transform
            res_gpu = self.fwd_tomo(u_gpu, 0)
            # copy result to cpu
            res[0,ids] = res_gpu.get()

        for k in range(self.nz//self.pnz):
            ids = np.arange(k*self.pnz, (k+1)*self.pnz)
            # copy data part to gpu
            u_gpu = cp.array(u[:,:,ids]).swapaxes(1,2)
            # Radon transform
            res_gpu = self.fwd_tomo(u_gpu, 0)
            # copy result to cpu
            res[1,ids] = res_gpu.get()
        return res
    
    def adj_tomo_batch(self, data):
        """Batch of adjoint Tomography transform (R*)"""
        res = np.zeros([3, self.nz, self.n, self.n], dtype='float32')
        for k in range(self.nz//self.pnz):
            ids = np.arange(k*self.pnz, (k+1)*self.pnz)
            # copy data part to gpu
            data_gpu = cp.array(data[0,ids])

            # Adjoint Radon transform
            res_gpu = self.adj_tomo(data_gpu, 0)
            
            # copy result to cpu
            res[:,ids] = res_gpu.get()
            
        for k in range(self.nz//self.pnz):
            ids = np.arange(k*self.pnz, (k+1)*self.pnz)
            # copy data part to gpu
            data_gpu = cp.array(data[1,ids])

            # Adjoint Radon transform
            res_gpu = self.adj_tomo(data_gpu, 0)
            
            # copy result to cpu
            res[:,:,ids] += res_gpu.swapaxes(1,2).get()
                        
        return res
    
    # @profile    
    def cg_tomo(self, xi0, u, titer, dbg):
        """CG solver for ||Ru-xi0||_2"""
        # minimization functional
        def minf(Ru):
            f = np.linalg.norm(Ru-xi0)**2
            return f
        for i in range(titer):
            Ru = self.fwd_tomo_batch(u)
            grad = self.adj_tomo_batch(Ru-xi0) / (self.ntheta * self.n/2)
            if i == 0:
                d = -grad
            else:
                # d = -grad+np.linalg.norm(grad)**2 / (np.sum(np.conj(d)*(grad-grad0))+1e-32)*d
                d = -grad+np.linalg.norm(grad)**2 / (np.sum(d*(grad-grad0))+1e-32)*d
            # line search
            # Rd = self.fwd_tomo_batch(d)
            gamma = 0.5#*self.line_search(minf, 1, Ru, Rd)
            grad0 = grad
            # update step
            u = u + gamma*d
            # check convergence
            if dbg and i%4==0:
                # import dxchange
                # dxchange.write_tiff(u[0,261].get(),f'rec/recc{i}',overwrite=True)
                residual = minf(Ru)
                np.save(f'residuals/r{i}',residual)
                print("%4d, %.3e, %.7e" %
                      (i, gamma, residual))
        return u

    def line_search(self, minf, gamma, Ru, Rd):
        """Line search for the step sizes gamma"""
        while (minf(Ru)-minf(Ru+gamma*Rd) < 0):
            gamma *= 0.5
        return gamma



    # batched versions of operators
    def fwd_tomo_small(self, u):
        res = cp.zeros([2,self.nz,self.ntheta,self.n], dtype='float32')
        res[0] = self.fwd_tomo(u, 0)
        res[1] = self.fwd_tomo(u.swapaxes(1,2), 0)            
        return res
    
    def adj_tomo_small(self, data):
        """Batch of adjoint Tomography transform (R*)"""
        res = self.adj_tomo(data[0], 0)
        res += self.adj_tomo(data[1], 0).swapaxes(1,2)                        
        return res
    
    def cg_tomo_small(self, xi0, u, titer, dbg):
        """CG solver for ||Ru-xi0||_2"""
        # minimization functional
        def minf(Ru):
            f = np.linalg.norm(Ru-xi0)**2
            return f
        for i in range(titer):
            Ru = self.fwd_tomo_small(u)
            grad = self.adj_tomo_small(Ru-xi0) / (self.ntheta * self.n/2)
            if i == 0:
                d = -grad
            else:
                # d = -grad+np.linalg.norm(grad)**2 / (np.sum(np.conj(d)*(grad-grad0))+1e-32)*d
                d = -grad+np.linalg.norm(grad)**2 / (np.sum(d*(grad-grad0))+1e-32)*d
            # line search
            # Rd = self.fwd_tomo_batch(d)
            gamma = 0.5#*self.line_search(minf, 1, Ru, Rd)
            grad0 = grad
            # update step
            u = u + gamma*d
            # check convergence
            if dbg:
                # import dxchange
                # dxchange.write_tiff(u[0,261].get(),f'rec/recc{i}',overwrite=True)
                residual = minf(Ru)
                np.save(f'residuals/r{i}',residual)
                print("%4d, %.3e, %.7e" %
                      (i, gamma, residual))
        return u

    # @profile    
    def cg_tomo(self, xi0, u, titer, dbg):
        """CG solver for ||Ru-xi0||_2"""
        # minimization functional
        def minf(Ru):
            f = np.linalg.norm(Ru-xi0)**2
            return f
        for i in range(titer):
            Ru = self.fwd_tomo_batch(u)
            grad = self.adj_tomo_batch(Ru-xi0) / (self.ntheta * self.n/2)
            if i == 0:
                d = -grad
            else:
                # d = -grad+np.linalg.norm(grad)**2 / (np.sum(np.conj(d)*(grad-grad0))+1e-32)*d
                d = -grad+np.linalg.norm(grad)**2 / (np.sum(d*(grad-grad0))+1e-32)*d
            # line search
            # Rd = self.fwd_tomo_batch(d)
            gamma = 0.5#*self.line_search(minf, 1, Ru, Rd)
            grad0 = grad
            # update step
            u = u + gamma*d
            # check convergence
            if dbg and i%4==0:
                # import dxchange
                # dxchange.write_tiff(u[0,261].get(),f'rec/recc{i}',overwrite=True)
                residual = minf(Ru)
                np.save(f'residuals/r{i}',residual)
                print("%4d, %.3e, %.7e" %
                      (i, gamma, residual))
        return u

    

    # # multi-gpu cg solver by slice partitions
    # def cg_tomo_multi_gpu(self, xi0, u, titer, lock, dbg, ids):

    #     global BUSYGPUS
    #     lock.acquire()  # will block if lock is already held
    #     for k in range(self.ngpus):
    #         if BUSYGPUS[k] == 0:
    #             BUSYGPUS[k] = 1
    #             gpu = k
    #             break
    #     lock.release()

    #     cp.cuda.Device(gpu).use()
    #     u_gpu = cp.array(u[:,ids])
    #     xi0_gpu = cp.array(xi0[:,ids])
    #     # reconstruct
    #     u_gpu = self.cg_tomo(xi0_gpu, u_gpu, titer, gpu, dbg)
    #     u[:,ids] = u_gpu.get()

    #     BUSYGPUS[gpu] = 0

    #     return u[:, ids]

    # def cg_tomo_batch(self, xi0, init, titer, dbg=False):
    #     """CG solver for rho||Ru-xi0||_2 by z-slice partitions"""
    #     u = init.copy()
    #     ids_list = [None]*int(np.ceil(self.nz/float(self.pnz)))
    #     for k in range(len(ids_list)):
    #         ids_list[k] = range(k*self.pnz, min(self.nz, (k+1)*self.pnz))

    #     lock = threading.Lock()
    #     global BUSYGPUS
    #     BUSYGPUS = np.zeros(self.ngpus)
    #     with cf.ThreadPoolExecutor(self.ngpus) as e:
    #         shift = 0
    #         for ui in e.map(partial(self.cg_tomo_multi_gpu, xi0, u, titer, lock, dbg), ids_list):
    #             u[:,np.arange(ui.shape[1])+shift] = ui
    #             shift += ui.shape[1]
    #     cp.cuda.Device(0).use()
    #     return u
