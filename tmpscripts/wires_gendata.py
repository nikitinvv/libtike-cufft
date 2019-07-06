import os
import signal
import sys

import cupy as cp
import dxchange
import numpy as np

import ptychotomo as pt

if __name__ == "__main__":

    igpu = np.int(sys.argv[1])
    cp.cuda.Device(igpu).use()  # gpu id to use
    # use cuda managed memory in cupy
    pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
    cp.cuda.set_allocator(pool.malloc)

    # Model parameters
    n = 64  # object size n x,y
    nz = 64  # object size in z
    ntheta = 400#3*n//8  # number of angles (rotations)
    voxelsize = 1e-6  # object voxel size
    energy = 5.0  # xray energy
    maxint = 0.1  # maximal probe intensity
    prbsize = 15  # probe size
    prbshift = 4  # probe shift (probe overlap = (1-prbshift)/prbsize)
    det = [63, 63]  # detector size
    noise = True  # apply discrete Poisson noise

    # Reconstrucion parameters
    model = 'gaussian'  # minimization funcitonal (poisson,gaussian)
    alpha = 1e-13  # tv regularization penalty coefficient
    piter = 4  # ptychography iterations
    titer = 4  # tomography iterations
    niter = 50  # ADMM iterations
    ptheta = ntheta  # number of angular partitions for simultaneous processing in ptychography
    pnz = nz  # number of slice partitions for simultaneous processing in tomography

    # Load a 3D object
    beta = dxchange.read_tiff('../data/test-beta-128.tiff')[::2,::2,::2]
    delta = dxchange.read_tiff('../data/test-delta-128.tiff')[::2,::2,::2]
    obj = cp.array(delta+1j*beta)

    # init probe, angles, scanner
    prb = cp.array(pt.probe(prbsize, maxint))
    theta = cp.linspace(0, 2*np.pi, ntheta).astype('float32')
    scan = cp.array(pt.scanner3(theta, obj.shape, prbshift,
                                prbshift, prbsize, spiral=1, randscan=False, save=False))
                            
    # Class gpu solver
    slv = pt.Solver(prb, scan, theta, det, voxelsize,
                    energy, ntheta, nz, n, ptheta, pnz)
    
    def signal_handler(sig, frame):  # Free gpu memory after SIGINT, SIGSTSTP
        slv = []
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTSTP, signal_handler)

    # Compute data
    data = slv.fwd_ptycho_batch(slv.exptomo(slv.fwd_tomo_batch(obj)))
    if (noise == True):  # Apply Poisson noise
        data = np.random.poisson(data).astype('float32')
    print("max intensity on the detector: ", np.sqrt(np.amax(data)))
    # Initial guess
    h = cp.zeros([ntheta, nz, n], dtype='complex64', order='C')+1
    psi = cp.zeros([ntheta, nz, n], dtype='complex64', order='C')+1
    #psi *= cp.exp(-1j*0.17)


    e = cp.zeros([3, nz, n, n], dtype='complex64', order='C')
    phi = cp.zeros([3, nz, n, n], dtype='complex64', order='C')
    lamd = cp.zeros([ntheta, nz, n], dtype='complex64', order='C')
    mu = cp.zeros([3, nz, n, n], dtype='complex64', order='C')
    u = cp.zeros([nz, n, n], dtype='complex64', order='C')

    # ADMM
    u, psi = slv.admm(data, h, e, psi, phi, lamd,
                            mu, u, alpha, piter, titer, niter, model)

    # Save result
    name = 'reg'+str(alpha)+'noise'+str(noise)+'maxint' + \
        str(maxint)+'prbshift'+str(prbshift)+'ntheta' + \
        str(ntheta)+str(model)+str(piter)+str(titer)+str(niter)

    dxchange.write_tiff(u.imag.get(),  'beta/beta'+name)
    dxchange.write_tiff(u.real.get(),  'delta/delta'+name)  # note sign change
    dxchange.write_tiff(cp.angle(psi).get(),  'psi/psiangle'+name)
    dxchange.write_tiff(cp.abs(psi).get(),  'psi/psiamp'+name)

    name = 'noise'+str(noise)+'prbshift'+str(prbshift)+'ntheta'+str(ntheta)

    np.save('/data/staff/tomograms/viknik/wires_data/data'+name,data)    
    np.save('/data/staff/tomograms/viknik/wires_data/coordinates'+name,scan.get())    
    np.save('/data/staff/tomograms/viknik/wires_data/theta'+name,theta.get())    
    np.save('/data/staff/tomograms/viknik/wires_data/prb'+name,prb.get())    
    