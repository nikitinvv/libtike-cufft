import os
import signal
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
    nscan = 1000 # number of scan positions
    nprb = 128  # probe size
    ndetx = 128
    ndety = 128
    
    
    # Reconstrucion parameterswe
    model = 'gaussian'  # minimization funcitonal (poisson,gaussian)
    piter = 100  # ptychography iterations
    ptheta = 1  # number of angular partitions for simultaneous processing in ptychography
    # prb = np.zeros([ntheta,prbsize,prbsize],dtype='complex64',order='C')
    # for k in range(0,ntheta):
    #     id = '%.3d' % (k+217)
    #     prbfile = h5py.File('/home/beams0/VNIKITIN/ptychotomo/Pillar_fly'+id+'_s128_i30_recon.h5','r')                
    #     prb[k] = prbfile['/probes/magnitude'][:prbsize,:].astype('float32')*np.exp(1j*prbfile['/probes/phase'][:prbsize,:].astype('float32'))        
    #     dxchange.write_tiff(np.abs(prb[k]),'prb217_amp',overwrite=True)
    #     dxchange.write_tiff(np.angle(prb[k]),'prb217_ang',overwrite=True)        
    # exit()
    prb = cp.ones([1,nprb,nprb],dtype='complex64')
    prb[0] = cp.array(dxchange.read_tiff('prb217_amp.tiff')*np.exp(1j*dxchange.read_tiff('prb217_ang.tiff')))   
    scan = cp.load('coords217.npy')[:,:,:nscan].astype('float32')    
    import matplotlib.pyplot as plt 
    plt.plot(scan[0,0,:].get(),scan[1,0,:].get(),'.',markersize=1.5,color='blue')
    plt.xlim([0,n])
    plt.ylim([0,nz])
    plt.savefig('fig3.png')
    
    maxint = cp.max(cp.abs(prb))
    # Class gpu solver
    slv = pt.Solver(maxint, nscan, nprb, ndetx, ndety, ntheta, nz, n, ptheta)

                   

    def signal_handler(sig, frame):  # Free gpu memory after SIGINT, SIGSTSTP
        slv = []
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTSTP, signal_handler)

    psi0 = cp.zeros([ntheta, nz, n], dtype='complex64', order='C')+1
    psi0[:,100:200,50:100]=1.2
    data = slv.fwd_ptycho_batch(psi0, scan, prb)
    dxchange.write_tiff(np.fft.fftshift(data),'data217')

    psi = cp.zeros([ntheta, nz, n], dtype='complex64', order='C')+0.3
    psi = slv.cg_ptycho_batch(data, psi, scan, prb, piter, model)    
    print("max intensity on the detector: ", np.amax(data))
        
    # Save result
    name = 'rec'+str(model)+str(piter)

    dxchange.write_tiff(cp.angle(psi).get(),  'psiangle'+name,overwrite=True)
    dxchange.write_tiff(cp.abs(psi).get(),  'psiamp'+name,overwrite=True)
