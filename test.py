import os
import signal
import sys

import cupy as cp
import dxchange
import numpy as np
import scipy
import ptychotomo as pt
import time
from skimage.restoration import unwrap_phase

if __name__ == "__main__":

    if (len(sys.argv) < 2):
        igpu = 0
    else:
        igpu = np.int(sys.argv[1])

    cp.cuda.Device(igpu).use()  # gpu id to use
    # use cuda managed memory in cupy
    pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
    cp.cuda.set_allocator(pool.malloc)

    # Model parameters
    prbsize = 256  # probe size
    prbshift = 8  # np.int(0.5667*113)
    det = [339, 339]  # detector size
    noise = True  # apply discrete Poisson noise
    model = 'gaussian'  # minimization funcitonal (poisson,gaussian)
    piter = 1024  # ptychography iterations
    pad = 128
    prbfile = 'probes-0.1'
    prbid = igpu+1
    amp0 = dxchange.read_tiff(
        'data/Cryptomeria_japonica-0256.tif').astype('float32')/255.0
    ang0 = dxchange.read_tiff(
        'data/Erdhummel_Bombus_terrestris-0256.tif').astype('float32')/255.0*np.pi
    prb = cp.array(np.load('probes/'+str(prbfile)+'.npy')[prbid].astype('complex64'))
    # prb = cp.abs(prb)**cp.exp(1j*cp.angle(prb))
    dxchange.write_tiff(ang0,  'anginit', overwrite=True)
    dxchange.write_tiff(amp0,  'ampinit', overwrite=True)
    # print(np.abs(prb).max())
    # exit()
    # build ps  i
    [nz, n] = np.shape(amp0)
    nz += 2*pad
    n += 2*pad
    amp = np.ones([nz, n], dtype=np.float32)*amp0[0, 0]
    ang = np.ones([nz, n], dtype=np.float32)*ang0[0, 0]
    amp[pad:-pad, pad:-pad] = amp0
    ang[pad:-pad, pad:-pad] = ang0    
    psi = amp*np.exp(1j*ang)    
    psi = cp.array(np.expand_dims(psi, axis=0))

    #dxchange.write_tiff(prb.get(),  'prb', overwrite=True)
    scan = cp.array(pt.scanner3(psi.shape, prbshift,
                                prbshift, prbsize, spiral=0, randscan=False, save=True))
    # Class gpu solver
    slv = pt.Solver(prb, scan, det, nz, n)

    def signal_handler(sig, frame):  # Free gpu memory after SIGINT, SIGSTSTP
        slv = []
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTSTP, signal_handler)

    # Compute data
    data = slv.fwd_ptycho_batch(psi)
    if (noise == True):  # Apply Poisson noise
        data = np.random.poisson(data).astype('float32')
    print('probe id '+str(prbid)+" max intensity on the detector: ", np.amax(data))

    # Initial guess
    init = cp.zeros([1, nz, n], dtype='complex64', order='C')+1 
    # CG scheme
    psi = slv.cg_ptycho_batch(data, init, piter, model)
    # Save result
    name = str(prbfile)+'prbid' + \
        str(prbid)+'prbshift'+str(prbshift)+str(model)
    dxchange.write_tiff(cp.angle(
        psi)[0, pad:-pad, pad:-pad].get(),  'psiang/psiang'+name, overwrite=True)
    dxchange.write_tiff(
        cp.abs(psi)[0, pad:-pad, pad:-pad].get(),  'psiamp/psiamp'+name, overwrite=True)
