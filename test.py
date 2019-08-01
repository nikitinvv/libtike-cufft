import os
import signal
import sys

import cupy as cp
import dxchange
import numpy as np
import scipy
import ptychocg as pt
import time
import h5py
if __name__ == "__main__":

    if (len(sys.argv) < 2):
        igpu = 0
    else:
        igpu = np.int(sys.argv[1])

    cp.cuda.Device(igpu).use()  # gpu id to use
    pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
    cp.cuda.set_allocator(pool.malloc)

    # Model parameters
    prbshiftx = 8  # 300/28.17822322520485
    prbshifty = prbshiftx*2/3.0
    n = np.int(prbshiftx*42+128-prbshiftx+1)
    nz = np.int(prbshifty*21+128-prbshifty+1)
    ntheta = 2
    prbsize = 128  # probe size
    det = [128, 128]  # detector size
    model = 'gaussian'  # minimization funcitonal (poisson,gaussian)
    piter = 100  # ptychography iterations

    # take probe
    prb0 = cp.array(dxchange.read_tiff('probereal128.tiff') + 1j *
                    dxchange.read_tiff('probeimag128.tiff'))[:128, :].astype('float32')
    prb = cp.zeros([ntheta, prbsize, prbsize], dtype='complex64')
    for k in range(ntheta):
        prb[k] = prb0  # +cp.random.random(prb0.shape)
    scan = cp.array(pt.scanner3([ntheta, nz, n], prbshiftx,
                                prbshifty, prbsize, spiral=0, randscan=False, save=True))
    prbmaxint = cp.max(cp.abs(prb))
    nscan = scan.shape[2]
    # Class gpu solver
    slv = pt.Solver(prbmaxint, nscan, prbsize, det, ntheta, nz, n)

    def signal_handler(sig, frame):  # Free gpu memory after SIGINT, SIGSTSTP
        slv = []
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTSTP, signal_handler)

    # gen synthetic data
    amp = dxchange.read_tiff(
        'data/Cryptomeria_japonica-1024.tif').astype('float32')[512:512+nz, 512:512+n]/255.0
    ang = dxchange.read_tiff(
        'data/Erdhummel_Bombus_terrestris-1024.tif').astype('float32')[512:512+nz, 512:512+n]/255.0*np.pi
    psi0 = cp.array(amp*np.exp(1j*ang))
    psi = cp.zeros([ntheta, nz, n], dtype='complex64')
    for k in range(ntheta):
        psi[k] = psi0*(k+1)
    data = slv.fwd_ptycho_batch(psi, scan, prb)
    print("max intensity on the detector: ", np.amax(data))

    # CG scheme
    init = cp.ones([ntheta, nz, n]).astype('complex64')*0.3
    psi = slv.cg_ptycho_batch(data, init, scan, prb, piter, model)

    # Save result
    name = str(model)
    dxchange.write_tiff(cp.angle(
        psi).get().astype('float32'),  'psiang/psiang'+name, overwrite=False)
    dxchange.write_tiff(
        cp.abs(psi).get().astype('float32'),  'psiamp/psiamp'+name)
