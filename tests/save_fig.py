import dxchange
import numpy as np
import tomovector as pt
import matplotlib.pyplot as plt
import time

if __name__ == "__main__":
    # f=plt.figure(figsize=(15,15)) #ROW,COLUMN
    # ux = dxchange.read_tiff('rec/datafourierrec.tiff')
    # sx = ux.shape[2]//2
    # sy = ux.shape[1]//2
    # sz = ux.shape[0]//2
    # print(ux.shape)
    # u = np.zeros([ux.shape[1]+ux.shape[0],ux.shape[1]+ux.shape[0]],dtype='float32')
    # u[:ux.shape[1],:ux.shape[2]] = ux[sz] 
    # u[:ux.shape[1],ux.shape[2]:] = ux[:,:,sx].swapaxes(0,1)
    # u[ux.shape[1]:,:ux.shape[2]] = ux[:,sy,:]
    # u[sy,:]=np.nan
    # u[:,sx]=np.nan
    # u[ux.shape[1]+sz,:]=np.nan
    # u[:,ux.shape[1]+sz]=np.nan
    # u[ux.shape[1],:]=np.min(u[~np.isnan(u)])
    # u[:,ux.shape[1]]=np.min(u[~np.isnan(u)])
    # plt.tick_params(axis='x', labelsize=24)
    # plt.tick_params(axis='y', labelsize=24)    
    # plt.imshow(u,cmap='gray',vmin=-200,vmax=200)
    # cb = plt.colorbar(fraction=0.046, pad=0.04)
    # cb.ax.tick_params(labelsize=24)
    # plt.xticks([0,100,200,300,400,500,600])
    # plt.yticks([0,100,200,300,400,500,600])
    # plt.xlabel('s',fontsize=24)
    # plt.ylabel('theta',fontsize=24)
    
    
    # plt.savefig('data.png')
    # plt.show()
    # exit()
    # Load object
    ux = dxchange.read_tiff('data/ux.tiff')[170:170+64]
    uy = dxchange.read_tiff('data/uy.tiff')[170:170+64]
    uz = dxchange.read_tiff('data/uz.tiff')[170:170+64]
    
    sx = 320
    sy = 320
    sz = 20
    urxf = dxchange.read_tiff('rec/urx_fourierrec.tiff')
    uryf = dxchange.read_tiff('rec/ury_fourierrec.tiff')
    urzf = dxchange.read_tiff('rec/urz_fourierrec.tiff')
    urxl = dxchange.read_tiff('rec/urx_linerec.tiff')
    uryl = dxchange.read_tiff('rec/ury_linerec.tiff')
    urzl = dxchange.read_tiff('rec/urz_linerec.tiff')
    
    u = np.zeros([ux.shape[1]+ux.shape[0],ux.shape[1]+ux.shape[0]],dtype='float32')
    u[:ux.shape[1],:ux.shape[2]] = ux[sz] 
    u[:ux.shape[1],ux.shape[2]:] = ux[:,:,sx].swapaxes(0,1)
    u[ux.shape[1]:,:ux.shape[2]] = ux[:,sy,:]
    u[sy,:]=np.nan
    u[:,sx]=np.nan
    u[ux.shape[1]+sz,:]=np.nan
    u[:,ux.shape[1]+sz]=np.nan
    u[ux.shape[1],:]=np.min(u[~np.isnan(u)])
    u[:,ux.shape[1]]=np.min(u[~np.isnan(u)])
    
    
    f=plt.figure(figsize=(35,25)) #ROW,COLUMN
    plt.subplot(231)
    plt.title('Ground truth',fontsize=32)
    plt.tick_params(axis='x', labelsize=24)
    plt.tick_params(axis='y', labelsize=24)    
    plt.ylabel('x component',rotation=90,fontsize=24)
    plt.imshow(u,cmap='gray',vmin = -0.8,vmax = 0.8)
    cb = plt.colorbar(fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=24)
    
    
    ux = urxf
    u = np.zeros([ux.shape[1]+ux.shape[0],ux.shape[1]+ux.shape[0]],dtype='float32')
    u[:ux.shape[1],:ux.shape[2]] = ux[sz] 
    u[:ux.shape[1],ux.shape[2]:] = ux[:,:,sx].swapaxes(0,1)
    u[ux.shape[1]:,:ux.shape[2]] = ux[:,sy,:]
    u[sy,:]=np.nan
    u[:,sx]=np.nan
    u[ux.shape[1]+sz,:]=np.nan
    u[:,ux.shape[1]+sz]=np.nan
    u[ux.shape[1],:]=np.min(u[~np.isnan(u)])
    u[:,ux.shape[1]]=np.min(u[~np.isnan(u)])
    plt.subplot(232)
    plt.tick_params(axis='x', labelsize=24)
    plt.tick_params(axis='y', labelsize=24)    
    plt.title('CG with Fourierrec',fontsize=32)
    plt.imshow(u,cmap='gray',vmin = -0.8,vmax = 0.8)
    cb = plt.colorbar(fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=24)

    ux = urxl
    u = np.zeros([ux.shape[1]+ux.shape[0],ux.shape[1]+ux.shape[0]],dtype='float32')
    u[:ux.shape[1],:ux.shape[2]] = ux[sz] 
    u[:ux.shape[1],ux.shape[2]:] = ux[:,:,sx].swapaxes(0,1)
    u[ux.shape[1]:,:ux.shape[2]] = ux[:,sy,:]
    u[sy,:]=np.nan
    u[:,sx]=np.nan
    u[ux.shape[1]+sz,:]=np.nan
    u[:,ux.shape[1]+sz]=np.nan
    u[ux.shape[1],:]=np.min(u[~np.isnan(u)])
    u[:,ux.shape[1]]=np.min(u[~np.isnan(u)])
    plt.subplot(233)
    plt.tick_params(axis='x', labelsize=24)
    plt.tick_params(axis='y', labelsize=24)    
    plt.title('CG with Linerec',fontsize=32)
    plt.imshow(u,cmap='gray',vmin = -0.8,vmax = 0.8)
    cb = plt.colorbar(fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=24)



    u = np.zeros([ux.shape[1]+ux.shape[0],ux.shape[1]+ux.shape[0]],dtype='float32')
    u[:ux.shape[1],:ux.shape[2]] = uy[sz] 
    u[:ux.shape[1],ux.shape[2]:] = uy[:,:,sx].swapaxes(0,1)
    u[ux.shape[1]:,:ux.shape[2]] = uy[:,sy,:]
    u[sy,:]=np.nan
    u[:,sx]=np.nan
    u[ux.shape[1]+sz,:]=np.nan
    u[:,ux.shape[1]+sz]=np.nan
    u[ux.shape[1],:]=np.min(u[~np.isnan(u)])
    u[:,ux.shape[1]]=np.min(u[~np.isnan(u)])
    
    
    plt.subplot(234)
    plt.tick_params(axis='x', labelsize=24)
    plt.tick_params(axis='y', labelsize=24)        
    plt.ylabel('y component',rotation=90,fontsize=24)
    plt.imshow(u,cmap='gray',vmin = -0.8,vmax = 0.8)
    cb = plt.colorbar(fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=24)
    
    
    uy = uryf
    u = np.zeros([ux.shape[1]+ux.shape[0],ux.shape[1]+ux.shape[0]],dtype='float32')
    u[:ux.shape[1],:ux.shape[2]] = uy[sz] 
    u[:ux.shape[1],ux.shape[2]:] = uy[:,:,sx].swapaxes(0,1)
    u[ux.shape[1]:,:ux.shape[2]] = uy[:,sy,:]
    u[sy,:]=np.nan
    u[:,sx]=np.nan
    u[ux.shape[1]+sz,:]=np.nan
    u[:,ux.shape[1]+sz]=np.nan
    u[ux.shape[1],:]=np.min(u[~np.isnan(u)])
    u[:,ux.shape[1]]=np.min(u[~np.isnan(u)])
    plt.subplot(235)
    plt.tick_params(axis='x', labelsize=24)
    plt.tick_params(axis='y', labelsize=24)    
    plt.imshow(u,cmap='gray',vmin = -0.8,vmax = 0.8)
    cb = plt.colorbar(fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=24)

    uy = uryl
    u = np.zeros([ux.shape[1]+ux.shape[0],ux.shape[1]+ux.shape[0]],dtype='float32')
    u[:ux.shape[1],:ux.shape[2]] = uy[sz] 
    u[:ux.shape[1],ux.shape[2]:] = uy[:,:,sx].swapaxes(0,1)
    u[ux.shape[1]:,:ux.shape[2]] = uy[:,sy,:]
    u[sy,:]=np.nan
    u[:,sx]=np.nan
    u[ux.shape[1]+sz,:]=np.nan
    u[:,ux.shape[1]+sz]=np.nan
    u[ux.shape[1],:]=np.min(u[~np.isnan(u)])
    u[:,ux.shape[1]]=np.min(u[~np.isnan(u)])
    plt.subplot(236)
    plt.tick_params(axis='x', labelsize=24)
    plt.tick_params(axis='y', labelsize=24)        
    plt.imshow(u,cmap='gray',vmin = -0.8,vmax = 0.8)
    
    cb = plt.colorbar(fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=24)



    plt.savefig('rec.png')
    plt.show()

    f=plt.figure(figsize=(15,10)) #ROW,COLUMN

    residuals = np.zeros(128)
    for i in range(128):
        residuals[i] = np.load(f'residuals_fourierrec/r{i}.npy')    
    plt.plot(residuals[:128],'r',label='fourierrec')
    plt.yscale("log")
    for i in range(128):
        residuals[i] = np.load(f'residuals_linerec/r{i}.npy')    
    plt.plot(residuals[:128],'b',label='linerec')
    plt.yscale("log")
    plt.tick_params(axis='x', labelsize=24)
    plt.ylabel('error',rotation=90,fontsize=24)
    plt.xlabel('iterations',fontsize=24)
    plt.tick_params(axis='y', labelsize=24)
    plt.legend(fontsize=24)    
    plt.grid()
    plt.savefig('convergence.png')
    plt.show()