import dxchange
import numpy as np
import cupy as cp
import tomovector as pt
import time

if __name__ == "__main__":

    Na = [256,512,1024,2048,4096]
    Nza = [256,512,256,64,16]  
    for k in range(len(Na)):
        N=Na[k]
        Nz=Nza[k]
        u = np.zeros([3,Nz,N,N],dtype='float32')
                
        # Init sizes and parameters                
        nz = u.shape[1] # vertical size
        n = u.shape[2] # horizontal size
        ntheta = N # number of projection angles         
        ngpus = 1 # number of gpus
        pnz = nz # chunk size to fit gpu memory
        
        
        # init angles
        theta = np.linspace(0,np.pi,ntheta).astype('float32')
        
        # init p
        p = np.zeros([3,1,ntheta,1],dtype='float32')
        p[0] = np.cos(theta)[np.newaxis,:,np.newaxis]
        p[1] = -np.sin(theta)[np.newaxis,:,np.newaxis]
        p[2] = 0# can be also used if needed
        
        method = 'linerec' # method for fwd and adj Radon transforms (fourierrec, linerec)
        with pt.SolverTomo(theta, p, ntheta, nz, n, pnz, n/2, method, ngpus) as slv:
            # generate data
            u = cp.array(u)
            data = slv.fwd_tomo(u)
            start_gpu = cp.cuda.Event()
            end_gpu = cp.cuda.Event()
            
            start_gpu.record()
            for k in range(3):
                data = slv.fwd_tomo(u)
            end_gpu.record()
            end_gpu.synchronize()        
            t_gpu1 = cp.cuda.get_elapsed_time(start_gpu, end_gpu)/3/1000
            
            
            start_gpu.record()
            for k in range(3):
                u = slv.adj_tomo(data)
            end_gpu.record()
            end_gpu.synchronize()        
            t_gpu2 = cp.cuda.get_elapsed_time(start_gpu, end_gpu)/3/1000
            print(f"{method} {N} {Nz} {t_gpu1:.2e} {t_gpu2:.2e} {(t_gpu1+t_gpu2):.2e} {((t_gpu1+t_gpu2)/Nz*N):.1e}")
            
        # method = 'linerec' # method for fwd and adj Radon transforms (fourierrec, linerec)
        # with pt.SolverTomo(theta, p, ntheta, nz, n, pnz, n/2, method, ngpus) as slv:
        #     # generate data
        #     u = cp.array(u)
        #     data = slv.fwd_tomo(u)
        #     start_gpu = cp.cuda.Event()
        #     end_gpu = cp.cuda.Event()
            
        #     start_gpu.record()
        #     for k in range(3):
        #         data = slv.fwd_tomo(u)
        #     end_gpu.record()
        #     end_gpu.synchronize()        
        #     t_gpu1 = cp.cuda.get_elapsed_time(start_gpu, end_gpu)/3/1000
            
            
        #     start_gpu.record()
        #     for k in range(3):
        #         u = slv.adj_tomo(data)
        #     end_gpu.record()
        #     end_gpu.synchronize()        
        #     t_gpu2 = cp.cuda.get_elapsed_time(start_gpu, end_gpu)/3/1000
        #     print(f"{method} {N} {Nz} {t_gpu1:.2e} {t_gpu2:.2e} {(t_gpu1+t_gpu2):.2e} {((t_gpu1+t_gpu2)/Nz*N):.1e}")
        