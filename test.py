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
    prbshift = 64  # np.int(0.5667*113)
    det = [339, 339]  # detector size
    noise = False  # apply discrete Poisson noise
    model = 'gaussian'  # minimization funcitonal (poisson,gaussian)
    piter = 2000  # ptychography iterations
    prbid = 1
    ampl0 = dxchange.read_tiff(
        'data/Cryptomeria_japonica-1024.tif')#[::4, ::4]
    angle0 = dxchange.read_tiff(
        'data/Erdhummel_Bombus_terrestris-1024.tif')#[::4, ::4]
    prb = cp.array(np.load('probes.npy')[prbid].astype('complex64'))

    ampl0 = ampl0/255.0
    angle0 = (angle0/255.0-0.5)*5#+np.pi/2

    pad = 128
    [nz, n] = np.shape(ampl0)
    nz+=2*pad
    n+=2*pad
    ampl = np.ones([nz, n], dtype=np.float32)*ampl0[0, 0]
    angle = np.ones([nz, n], dtype=np.float32)*angle0[0, 0]
    ampl[pad:-pad, pad:-pad] = ampl0
    angle[pad:-pad, pad:-pad] = angle0
    # build psi
    psi = ampl*np.exp(1j*angle)

    # save
    dxchange.write_tiff(np.angle(psi),  'psianginit', overwrite=True)
    dxchange.write_tiff(np.abs(psi),  'psiampinit', overwrite=True)
    psi = cp.array(np.expand_dims(psi, axis=0))

    
    dxchange.write_tiff(prb.get(),  'prb', overwrite=True)
    scan = cp.array(pt.scanner3(psi.shape, prbshift,
                                prbshift, prbsize, spiral=0, randscan=False, save=True))
    print(scan.shape)
    # # Class gpu solver
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
    print(str(prbid)+" max intensity on the detector: ", np.amax(data))

    # Initial guess
    init = cp.zeros([1, nz, n], dtype='complex64', order='C')+1#*cp.exp(1j*1)

    # CG scheme
    psi = slv.cg_ptycho_batch(data, init, piter, model)
    
    anglrec = cp.angle(psi).get()#-cp.mean(cp.angle(psi[0, pad/4:3*pad/4, n/2-pad/4:n/2+pad/4]))).get()
    amprec = cp.abs(psi).get()
    anglrecu = unwrap_phase(anglrec).astype('float32')
    # anglrec[anglrec>np.pi]-=2*np.pi
    # anglrec[anglrec<-np.pi]+=2*np.pi
    print(np.max(anglrec))
    print(np.min(anglrec))
    # Save result
    name = 'noise'+str(noise)+'prbid' + \
        str(prbid)+'prbshift'+str(prbshift)+str(model)+str(piter)
    dxchange.write_tiff(anglrec[0,pad:-pad,pad:-pad],  'psiangle/psiangle'+name, overwrite=True)
    dxchange.write_tiff(anglrecu[0,pad:-pad,pad:-pad],  'psiangle/psiangleu'+name, overwrite=True)
    dxchange.write_tiff(amprec[0,pad:-pad,pad:-pad], 'psiamp/psiamp'+name,overwrite=True)
    