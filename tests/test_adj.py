import tomopy
import dxchange
import numpy as np
import tomovector as pt
import cupy as cp

# tomopy
objx = dxchange.read_tiff('data/phantom_00012/M4R1_mx.tif').astype('float32')
objy = dxchange.read_tiff('data/phantom_00012/M4R1_my.tif').astype('float32')
objz = dxchange.read_tiff('data/phantom_00012/M4R1_mz.tif').astype('float32')

npad = ((234, 234), (116, 116), (52, 52))
objx = np.pad(objx, npad, mode='constant', constant_values=0)
objy = np.pad(objy, npad, mode='constant', constant_values=0)
objz = np.pad(objz, npad, mode='constant', constant_values=0)

theta = tomopy.angles(31, 90, 270)
# prj1 = tomopy.project3(objx, objy, objz, theta, axis=0, pad=False)
# dxchange.write_tiff(prj1, 'data/prj_tomopy', overwrite=True)

# save padded object for further use
dxchange.write_tiff(objx,'data/ux.tiff',overwrite=True)
dxchange.write_tiff(objy,'data/uy.tiff',overwrite=True)
dxchange.write_tiff(objz,'data/uz.tiff',overwrite=True)

obj = np.zeros([3, *objx.shape], dtype='float32')
obj[0] = np.array(objx)
obj[1] = np.array(objy)
obj[2] = np.array(objz)

# init variables
ntheta = len(theta)
nz = obj.shape[1]
n = obj.shape[-1]
pnz = nz//4
center = n/2
ngpus = 1
theta = theta.astype('float32')

# p
p = np.zeros([3, 1, ntheta, 1], dtype='float32')
p[0] = np.cos(theta)[np.newaxis, :, np.newaxis]
p[1] = -np.sin(theta)[np.newaxis, :, np.newaxis]
p[2] = 0

method = 'fourierrec'
with pt.SolverTomo(theta, p, ntheta, nz, n, pnz, center, method,  ngpus) as slv:
    # FWD operator
    prj = slv.fwd_tomo_batch(obj)

    # ADJ operator
    aobj = slv.adj_tomo_batch(prj)

    dxchange.write_tiff(prj, f'data/prj{method}', overwrite=True)
    dxchange.write_tiff(aobj[0], f'data/aobj{method}', overwrite=True)

    t1 = np.sum(prj*np.conj(prj))
    t2 = np.sum(aobj*np.conj(obj))
    print(f'Adjoint test for method {method}:{t1} ?= {t2}')

method = 'linerec'
with pt.SolverTomo(theta, p, ntheta, nz, n, pnz, center, method, ngpus) as slv:
    # FWD operator
    prj = slv.fwd_tomo_batch(obj)

    # ADJ operator
    aobj = slv.adj_tomo_batch(prj)

    dxchange.write_tiff(prj, f'data/prj{method}', overwrite=True)
    dxchange.write_tiff(aobj[0], f'data/aobj{method}', overwrite=True)

    t1 = np.sum(prj*np.conj(prj))
    t2 = np.sum(aobj*np.conj(obj))
    print(f'Adjoint test for method {method}:{t1} ?= {t2}')
