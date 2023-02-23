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
    view = 4
    ux = dxchange.read_tiff('data/ux.tiff')[170:170+64]
    uy = dxchange.read_tiff('data/uy.tiff')[170:170+64]
    uz = dxchange.read_tiff('data/uz.tiff')[170:170+64]
    ux0=ux.copy()
    uy0=uy.copy()
    uz0=uz.copy()

    
    sx = 384
    sy = 384
    sz = 32
    urxf = dxchange.read_tiff(f'rec/urx_{view}_fourierrec.tiff')
    uryf = dxchange.read_tiff(f'rec/ury_{view}_fourierrec.tiff')
    urzf = dxchange.read_tiff(f'rec/urz_{view}_fourierrec.tiff')
    
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
    plt.title(f'{view} view',fontsize=32)
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
    
    
    
    
    
    
    
    
    
    
    
    
    

    ux = urxf-ux0
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
    plt.title(f'{view} view, l2 difference {np.linalg.norm(ux)/np.linalg.norm(ux0):0.7f}',fontsize=32)
    plt.imshow(u,cmap='gray',vmin = -0.8/100,vmax = 0.8/100)
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
    
    
    plt.subplot(236)
    plt.tick_params(axis='x', labelsize=24)
    plt.tick_params(axis='y', labelsize=24)        
    plt.ylabel('y component',rotation=90,fontsize=24)
    plt.imshow(u,cmap='gray',vmin = -0.8,vmax = 0.8)
    cb = plt.colorbar(fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=24)
    
    
    uy = uryf-uy0
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
    plt.imshow(u,cmap='gray',vmin = -0.8/100,vmax = 0.8/100)
    cb = plt.colorbar(fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=24)











    plt.savefig(f'rec{view}.png')
    plt.show()
