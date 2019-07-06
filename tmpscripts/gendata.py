import ptychotomo as pt
import dxchange
import numpy as np
import cupy as cp
import signal
import sys
import os
import sys
import time

if __name__ == "__main__":
    
    igpu = np.int(sys.argv[1])    
    cp.cuda.Device(igpu).use()  # gpu id to use
    idd = np.int(sys.argv[2])
    print("gpu id:",igpu)
    
    pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
    cp.cuda.set_allocator(pool.malloc)
    # Model parameters
    n = 512
    nz = 192
    voxelsize = 1e-6/2  # object voxel size
    energy = 8.8  # xray energy
    maxint = [3,0.3,0.03,0.003][igpu]
    alpha = [8e-9,1e-8,2e-8,4e-8][igpu]
    
    
    prbsize = 16 # probe size
    prbshift = 8  # probe shift (probe overlap = (1-prbshift)/prbsize)
    det = [128, 128] # detector size
    ntheta = n*3//4  # number of angles (rotations)
    noise = True  # apply discrete Poisson noise
    
    ptheta = 2
    pnz = 8
    beta = dxchange.read_tiff('../data/beta192.tiff')
    delta = -dxchange.read_tiff('../data/delta192.tiff')
    obj = cp.array(delta+1j*beta)
    

    prb = cp.array(pt.objects.probe(prbsize, maxint))
    theta = cp.linspace(0, np.pi, ntheta).astype('float32')
    scan = cp.array(pt.objects.scanner3(theta, obj.shape, prbshift,
                                    prbshift, prbsize, spiral=0, randscan=True, save=False)) 
    #tomoshape = [len(theta), obj.shape[0], obj.shape[2]]
    # Class gpu solver 
    slv = pt.solver.Solver(prb, scan, theta, det, voxelsize, energy, ntheta, nz, n, ptheta, pnz)
    # Free gpu memory after SIGINT, SIGSTSTP
    def signal_handler(sig, frame):
        slv = []
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTSTP, signal_handler)

    # Compute data
    psi = slv.exptomo(slv.fwd_tomo_batch(obj))
    
    # Compute data
    data = slv.fwd_ptycho_batch(slv.exptomo(slv.fwd_tomo_batch(obj)))
    if (noise == True):  # Apply Poisson noise
        for k in range(0,data.shape[0]):    
            data[k] = np.random.poisson(data[k]).astype('float32')
    print("max intensity on the detector: ", np.amax(data))


    name = str(idd)+'noise'+str(noise)+'maxint' + \
        str(maxint)+'prbshift'+str(prbshift)+'ntheta'+str(ntheta)


    np.save('/data/staff/tomograms/viknik/gendata/data'+name,data)    
    np.save('/data/staff/tomograms/viknik/gendata/coordinates'+name,scan.get())    
    np.save('/data/staff/tomograms/viknik/gendata/theta'+name,theta.get())    
    np.save('/data/staff/tomograms/viknik/gendata/prb'+name,prb.get())    
    
       
