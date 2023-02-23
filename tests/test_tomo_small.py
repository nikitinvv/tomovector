import dxchange
import numpy as np
import tomovector as pt
import time
import scipy.ndimage as ndimage 
import cupy as cp
if __name__ == "__main__":

    # Load object
    # Load object
    ux = dxchange.read_tiff('rec/ux.tiff')
    uy = dxchange.read_tiff('rec/uy.tiff')
    uz = dxchange.read_tiff('rec/uz.tiff')
    u = np.zeros([3,*ux.shape],dtype='float32')
    u[0] = ux
    u[1] = uy
    u[2] = uz
    u = u[:,64:-64,64:-64,64:-64]
    # u = u[:,::2,::2,::2]
    u = ndimage.gaussian_filter(u, sigma=2)
    
    dxchange.write_tiff(u[0],'rec/ux.tiff',overwrite=True)
    dxchange.write_tiff(u[1],'rec/uy.tiff',overwrite=True)
    dxchange.write_tiff(u[2],'rec/uz.tiff',overwrite=True)
    # exit()
    print(u.shape)
    # Init sizes and parameters                    
    nz = u.shape[1] # vertical size
    n = u.shape[2] # horizontal size
    ntheta = 640 # number of projection angles         
    ngpus = 1 # number of gpus
    pnz = nz # chunk size to fit gpu memory
    niter = 32 # number of iterations
    method = 'fourierrec' # method for fwd and adj Radon transforms (fourierrec, linerec)
    dbg = True # show convergence
    
    p = np.zeros([3,ntheta,1],dtype='float32')
    # init angles
    theta = np.linspace(0,np.pi,ntheta).astype('float32')
        
    p[0] = np.cos(theta)[np.newaxis,:,np.newaxis]
    p[1] = -np.sin(theta)[np.newaxis,:,np.newaxis]
    p[2] = 0
        
    with pt.SolverTomo(theta, p, ntheta, nz, n, pnz, n/2, method, ngpus) as slv:
        # generate data        
        u = cp.array(u)
        data = slv.fwd_tomo_small(u)
        noise = cp.abs(data)-cp.random.poisson(cp.abs(data))
        data+=noise
        dxchange.write_tiff(data.get(),  f'rec/data_noise_{method}', overwrite=True)
        # exit()
        # data[:]+=noise
        # dxchange.write_tiff(data,  f'rec/datanoise{method}', overwrite=True)
        # exit()
        # CG sovler
        t = time.time()
        ur = slv.cg_tomo_small(data, u*0, niter, dbg)
        print(f'Rec time {time.time()-t}s')
    # save results
    dxchange.write_tiff(ur[0].get(),  f'rec/urx_n_2_{method}', overwrite=True)
    dxchange.write_tiff(ur[1].get(),  f'rec/ury_n_2_{method}', overwrite=True)
    dxchange.write_tiff(ur[2].get(),  f'rec/urz_n_2_{method}', overwrite=True)
    