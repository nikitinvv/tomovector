

from tomo import FourierRec
import cupyx.scipy.ndimage as ndimage
import cupy as cp
import tifffile

n = 256
nz = 16 # at least 2 
ntheta = 3*n//2
center = n/2

theta = cp.linspace(0,cp.pi,ntheta,endpoint=True).astype('float32')

# object
f = tifffile.imread('chip.tiff').astype('float32')
f = cp.array(f[128-nz//2:128+nz//2])
f *= 300 # vary attenuation

# some flat,dark fields
flat = 10000+cp.zeros([nz,1,n],dtype='float32')
dark = cp.zeros([nz,1,n],dtype='float32')

with FourierRec(n, ntheta, nz, theta, center) as cl:    
    # generate data exp(-Rf)*(flat-dark)+dark   
    data = cp.exp(-cl.fwd(f))*(flat-dark)+dark     
    # apply noise
    data = cp.random.poisson(data.astype('int32')).astype('float32')
    # reconstruct        
    g = -cp.log((data-dark)/(flat-dark))
    # fbp filter
    g = cl.fbp_filter(g,'ramp')#parzen,shepp
    # backprojection      
    ff = cl.adj(g)    
    
tifffile.imwrite('data.tiff',data.get().swapaxes(0,1))    
tifffile.imwrite('rec.tiff',ff.get())    
