

from tomo import FourierRec
import cupy as cp
import tifffile

n = 256
nz = 32 # at least 2 
ntheta = 3*n//2
center = n/2

theta = cp.linspace(0,cp.pi,ntheta,endpoint=True).astype('float32')
f = cp.zeros([nz,n,n],dtype='float32')
f[:,n/2-n//4:n/2+n//4,n/2-n//4:n/2+n//4] = 1

with FourierRec(n, ntheta, nz, theta, center) as cl:
    g = cl.fwd(f)    
    ff = cl.adj(g)    

print(f'{cp.sum(g*g)} ?={cp.sum(ff*f)}')
    
tifffile.imwrite('data.tiff',g.get())    
