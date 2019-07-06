import ptychotomo as pt
import dxchange
import numpy as np
import cupy as cp
import signal
import sys
import os
import sys


if __name__ == "__main__":
    igpu = np.int(sys.argv[1])
    idd = np.int(sys.argv[2])
    cp.cuda.Device(igpu).use()  # gpu id to use    
    print("gpu id:",igpu)


    pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
    cp.cuda.set_allocator(pool.malloc)
    # Model parameters
    voxelsize = 1e-6/2  # object voxel size
    energy = 8.8  # xray energy
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
    ntheta = n*3//4
    noise = True
    # Reconstrucion parameters
    model = 'poisson'  # minimization funcitonal (poisson,gaussian)
    piter = 4  # ptychography iterations
    titer = 4  # tomography iterations
    NITER = 300  # ADMM iterations

    ptheta = 1
    pnz = 8
    
    rho = 0.5
    tau = 0.5

    name = str(idd)+'noise'+str(noise)+'maxint' + \
        str(maxint)+'prbshift'+str(prbshift)+'ntheta'+str(ntheta)

    scan = cp.array(np.load('/data/staff/tomograms/viknik/gendata/coordinates'+name+'.npy'))
    data = np.load('/data/staff/tomograms/viknik/gendata/data'+name+'.npy')
    prb = cp.array(np.load('/data/staff/tomograms/viknik/gendata/prb'+name+'.npy'))
    theta = cp.array(np.load('/data/staff/tomograms/viknik/gendata/theta'+name+'.npy'))

    print("max data = ", np.amax(data))      

    # Class gpu solver 
    slv = pt.solver.Solver(prb, scan, theta, det, voxelsize, energy, ntheta, nz, n, ptheta, pnz)
    # Free gpu memory after SIGINT, SIGSTSTP
    def signal_handler(sig, frame):
        slv = []
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTSTP, signal_handler)



    # Initial guess
    h = cp.zeros([ntheta, nz, n], dtype='complex64', order='C')+1
    psi = cp.zeros([ntheta, nz, n], dtype='complex64', order='C')+1
    e = cp.zeros([3, nz, n, n], dtype='complex64', order='C')
    phi = cp.zeros([3, nz, n, n], dtype='complex64', order='C')
    lamd = cp.zeros([ntheta, nz, n], dtype='complex64', order='C')
    mu = cp.zeros([3, nz, n, n], dtype='complex64', order='C')
    u = cp.zeros([nz, n, n], dtype='complex64', order='C')
    
    # ADMM
    u, psi = slv.admm(data, h, e, psi, phi, lamd,
                            mu, u, alpha, rho, tau, piter, titer, NITER, model,idd)
    # Save result
    name = str(idd)+'reg'+str(alpha)+'noise'+str(noise)+'maxint' + \
        str(maxint)+'prbshift'+str(prbshift)+'ntheta'+str(ntheta)+str(model)+str(piter)+str(titer)+str(NITER)

    dxchange.write_tiff(u.imag.get(),  '/data/staff/tomograms/viknik/beta/beta_'+name)
    dxchange.write_tiff(u.real.get(),  '/data/staff/tomograms/viknik/delta/delta_'+name)
    # if not os.path.exists('/data/staff/tomograms/viknik/lagr'):
    #     os.makedirs('/data/staff/tomograms/viknik/lagr')
    # np.save('/data/staff/tomograms/viknik/lagr/lagr'+name,lagr.get())
    


