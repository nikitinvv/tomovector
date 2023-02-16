import dxchange
import numpy as np
import tomovector as pt
import time

if __name__ == "__main__":

    # Load object
    ux = dxchange.read_tiff('data/ux.tiff')
    uy = dxchange.read_tiff('data/uy.tiff')
    uz = dxchange.read_tiff('data/uz.tiff')
    u = np.zeros([3,*ux.shape],dtype='float32')
    u[0] = ux
    u[1] = uy
    u[2] = uz
            
    # Init sizes and parameters                
    nz = u.shape[1] # vertical size
    n = u.shape[2] # horizontal size
    ntheta = 512 # number of projection angles         
    ngpus = 1 # number of gpus
    pnz = nz # chunk size to fit gpu memory
    niter = 64 # number of iterations
    method = 'linerec' # method for fwd and adj Radon transforms (fourierrec, linerec)
    dbg = True # show convergence
    
    # init angles
    theta = np.linspace(0,np.pi,ntheta).astype('float32')
    
    # init p
    p = np.zeros([3,1,ntheta,1],dtype='float32')
    p[0] = np.cos(theta)[np.newaxis,:,np.newaxis]
    p[1] = -np.sin(theta)[np.newaxis,:,np.newaxis]
    p[2] = 0# can be also used if needed
    with pt.SolverTomo(theta, p, ntheta, nz, n, pnz, n/2, method, ngpus) as slv:
        # generate data
        data = slv.fwd_tomo_batch(u)
        dxchange.write_tiff(data,  f'rec/data{method}', overwrite=True)
        # CG sovler
        t = time.time()
        ur = slv.cg_tomo_batch(data, u*0, niter, dbg)
        print(f'Rec time {time.time()-t}s')
    # save results
    dxchange.write_tiff(ur[0],  f'rec/urx_{method}', overwrite=True)
    dxchange.write_tiff(ur[1],  f'rec/ury_{method}', overwrite=True)
    dxchange.write_tiff(ur[2],  f'rec/urz_{method}', overwrite=True)
    