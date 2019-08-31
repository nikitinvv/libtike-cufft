import os
import sys
import cupy as cp
import dxchange
import numpy as np
import h5py
import ptychocg as pt

if __name__ == "__main__":

    if (len(sys.argv) < 2):
        igpu = 0
    else:
        igpu = np.int(sys.argv[1])
    cp.cuda.Device(igpu).use()  # gpu id to use

    # set cupy to use unified memory
    pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
    cp.cuda.set_allocator(pool.malloc)

    # sizes
    n = 600 # horizontal size
    nz = 276 # vertical size
    ntheta = 1  # number of angles (rotations)
    nscan = 100 # number of scan positions
    nprb = 128  # probe size
    ndetx = 128 # detector x size
    ndety = 128 # detector y size
        
    # Reconstrucion parameters
    model = 'gaussian'  # minimization funcitonal (poisson,gaussian)
    piter = 100  # ptychography iterations
    ptheta = 1  # number of angular partitions for simultaneous processing in ptychography

    # read correct probe
    prb = cp.ones([ntheta,nprb,nprb],dtype='complex64')
    prbamp = cp.array(dxchange.read_tiff('test/prb217_amp.tiff').astype('float32'))
    prbang = cp.array(dxchange.read_tiff('test/prb217_ang.tiff').astype('float32'))
    prb[0] = prbamp*cp.exp(1j*prbang)    
    
    # read scan positions
    scan = cp.ones([2,ntheta,nscan],dtype='float32')
    scan[:,0] = cp.load('test/coords217.npy')[:,:nscan].astype('float32')            

    # read object
    psi0 = cp.zeros([ntheta,nz, n], dtype='complex64', order='C')+1
    psiamp = cp.array(dxchange.read_tiff('test/initpsiamp.tiff').astype('float32'))
    psiang = cp.array(dxchange.read_tiff('test/initpsiang.tiff').astype('float32'))
    psi0[0] = psiamp*cp.exp(1j*psiang)    

    # Class gpu solver
    slv = pt.Solver(nscan, nprb, ndetx, ndety, ntheta, nz, n, ptheta)        
    # Compute data
    data = slv.fwd_ptycho_batch(psi0, scan, prb)
    dxchange.write_tiff(np.fft.fftshift(data),'test/data217',overwrite=True)

    # Initial guess
    psi = cp.zeros([ntheta, nz, n], dtype='complex64', order='C')+1    
    prb = prb.swapaxes(1,2)

    # CG solver
    psi, prb = slv.cg_ptycho_batch(data, psi, scan, prb, piter, model)                
    # Save result
    name = 'rec'+str(model)+str(piter)    
    dxchange.write_tiff(cp.angle(psi).get(),  'test/psiangle'+name,overwrite=True)
    dxchange.write_tiff(cp.abs(psi).get(),  'test/psiamp'+name,overwrite=True)
    dxchange.write_tiff(cp.angle(prb).get(),  'test/prbangle'+name,overwrite=True)
    dxchange.write_tiff(cp.abs(prb).get(),  'test/prbamp'+name,overwrite=True)    
    
    import matplotlib.pyplot as plt 
    plt.plot(scan[0,0,:].get(),scan[1,0,:].get(),'.',markersize=1.5,color='blue')
    plt.xlim([0,n])
    plt.ylim([0,nz])  
    plt.savefig('test/scan.png')  
    plt.subplot(2,2,1)
    plt.imshow(cp.angle(psi[0]).get(),cmap='gray')
    plt.subplot(2,2,2)
    plt.imshow(cp.abs(psi[0]).get(),cmap='gray')
    plt.subplot(2,2,3)
    plt.imshow(cp.angle(prb[0]).get(),cmap='gray')
    plt.subplot(2,2,4)
    plt.imshow(cp.abs(prb[0]).get(),cmap='gray')        
    plt.savefig('test/result.png',dpi=300)
    












    # prb = np.zeros([ntheta,prbsize,prbsize],dtype='complex64',order='C')
    # for k in range(0,ntheta):
    #     id = '%.3d' % (k+217)
    #     prbfile = h5py.File('/home/beams0/VNIKITIN/ptychotomo/Pillar_fly'+id+'_s128_i30_recon.h5','r')                
    #     prb[k] = prbfile['/probes/magnitude'][:prbsize,:].astype('float32')*np.exp(1j*prbfile['/probes/phase'][:prbsize,:].astype('float32'))        
    #     dxchange.write_tiff(np.abs(prb[k]),'prb217_amp',overwrite=True)
    #     dxchange.write_tiff(np.angle(prb[k]),'prb217_ang',overwrite=True)        
    # exit()