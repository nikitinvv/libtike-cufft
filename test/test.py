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
    pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
    cp.cuda.set_allocator(pool.malloc)

    n = 600
    nz = 276
    ntheta = 1  # number of angles (rotations)
    nscan = 2000 # number of scan positions
    nprb = 128  # probe size
    ndetx = 128
    ndety = 128
    
    
    # Reconstrucion parameterswe
    model = 'gaussian'  # minimization funcitonal (poisson,gaussian)
    piter = 128  # ptychography iterations
    ptheta = 1  # number of angular partitions for simultaneous processing in ptychography

    prb = cp.ones([1,nprb,nprb],dtype='complex64')
    prb[0] = cp.array(dxchange.read_tiff('test/prb217_amp.tiff')*np.exp(1j*dxchange.read_tiff('test/prb217_ang.tiff')))
    scan = cp.load('test/coords217.npy')[:,:,:nscan].astype('float32')    
    import matplotlib.pyplot as plt 
    plt.plot(scan[0,0,:].get(),scan[1,0,:].get(),'.',markersize=1.5,color='blue')
    plt.xlim([0,n])
    plt.ylim([0,nz])
    plt.savefig('fig3.png')
    
    maxint = cp.max(cp.abs(prb))
    # Class gpu solver
    slv = pt.Solver(maxint, nscan, nprb, ndetx, ndety, ntheta, nz, n, ptheta)

    
    psi0 = cp.zeros([ntheta, nz, n], dtype='complex64', order='C')+1
    psi0[0] = cp.array(dxchange.read_tiff('test/Cryptomeria_japonica-1024.tif')[200:200+nz,200:200+n]/255.0*np.exp(1j*dxchange.read_tiff('test/Erdhummel_Bombus_terrestris-1024.tif')[200:200+nz,200:200+n]/255.0))    
    data = slv.fwd_ptycho_batch(psi0, scan, prb)
    dxchange.write_tiff(np.fft.fftshift(data),'test/data217')

    psi = cp.zeros([ntheta, nz, n], dtype='complex64', order='C')+1
    
    # prb = cp.zeros([ntheta, nprb, nprb], dtype='complex64', order='C')+1
    # prb+=cp.random.random([ntheta,nprb,nprb])*0.2
    prb=prb.swapaxes(1,2)
    psi,prb = slv.cg_ptycho_batch(data, psi, scan, prb, piter, model)    
    print("max intensity on the detector: ", np.amax(data))
        
    # Save result
    name = 'rec'+str(model)+str(piter)
    
    dxchange.write_tiff(cp.angle(psi).get(),  'psiangle'+name,overwrite=True)
    dxchange.write_tiff(cp.abs(psi).get(),  'psiamp'+name,overwrite=True)
    dxchange.write_tiff(cp.angle(prb).get(),  'prbangle'+name,overwrite=True)
    dxchange.write_tiff(cp.abs(prb).get(),  'prbamp'+name,overwrite=True)    

















    # prb = np.zeros([ntheta,prbsize,prbsize],dtype='complex64',order='C')
    # for k in range(0,ntheta):
    #     id = '%.3d' % (k+217)
    #     prbfile = h5py.File('/home/beams0/VNIKITIN/ptychotomo/Pillar_fly'+id+'_s128_i30_recon.h5','r')                
    #     prb[k] = prbfile['/probes/magnitude'][:prbsize,:].astype('float32')*np.exp(1j*prbfile['/probes/phase'][:prbsize,:].astype('float32'))        
    #     dxchange.write_tiff(np.abs(prb[k]),'prb217_amp',overwrite=True)
    #     dxchange.write_tiff(np.angle(prb[k]),'prb217_ang',overwrite=True)        
    # exit()