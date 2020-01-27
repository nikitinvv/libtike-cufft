import sys

import dxchange
import cupy as cp
import numpy as np
from scipy import ndimage

import libtike.cufft as pt

if __name__ == "__main__":

    if (len(sys.argv) < 2):
        igpu = 0
    else:
        igpu = np.int(sys.argv[1])

    # sizes
    n = 600  # horizontal size
    nz = 276  # vertical size
    ntheta = 1  # number of projections
    nscan = 1100  # number of scan positions [max 5706 for the data example]
    nprb = 128  # probe size
    ndet = 128  # detector x size
    recover_prb = True  # True: recover probe, False: use the initial one
    # Reconstrucion parameters
    model = 'gaussian'  # minimization funcitonal (poisson,gaussian)
    piter = 128 # ptychography iterations
    ptheta = 1  # number of angular partitions for simultaneous processing in ptychography    
    nmodes = 3 # number of probe modes for decomposition in reconstruction
    # read real probe
    prb_real = np.zeros([ntheta,nmodes, nprb, nprb], dtype='complex64')    
    prb_amp_real = dxchange.read_tiff('model/probes_amp_real.tiff')[0:nmodes].astype('float32')
    prb_ang_real = dxchange.read_tiff('model/probes_ang_real.tiff')[0:nmodes].astype('float32')
    prb_real[0] = prb_amp_real*np.exp(1j*prb_ang_real)
    # read initial guess for the probe
    prb_init = np.zeros([ntheta,nmodes, nprb, nprb], dtype='complex64')
    prb_amp_init = dxchange.read_tiff('model/probes_amp.tiff')[0:nmodes].astype('float32')
    prb_ang_init = dxchange.read_tiff('model/probes_ang.tiff')[0:nmodes].astype('float32')
    prb_init[0] = prb_amp_init*np.exp(1j*prb_ang_init)

    for k in range(nmodes):        
        prb_init[:,k]/=np.max(np.abs(prb_init[:,k]))
    # read scan positions
    scan = np.ones([ntheta, nscan, 2], dtype='float32')
    temp = np.moveaxis(np.load('model/coords.npy'), 0, 1)[:nscan*5:5]
    scan[0, :, 0] = temp[:, 1]
    scan[0, :, 1] = temp[:, 0]

    # read object
    psi0 = np.ones([ntheta, nz, n], dtype='complex64')
    psiamp = dxchange.read_tiff('model/initpsiamp.tiff').astype('float32')
    psiang = dxchange.read_tiff('model/initpsiang.tiff').astype('float32')
    psi0[0] = psiamp*np.exp(1j*psiang)

    # Class gpu solver
    with pt.CGPtychoSolver(nscan, nprb, ndet, ptheta, nz, n) as slv:
        # Compute intensity data on the detector |FQ|**2
        data = np.zeros([ntheta,nscan,ndet,ndet],dtype='float32')
        for k in range(nmodes):
            data += np.abs(slv.fwd_ptycho_batch(psi0, scan, prb_real[:,k]))**2
        dxchange.write_tiff(data, 'data', overwrite=True)
        
        # Initial guess
        psi = np.ones([ntheta, nz, n], dtype='complex64')
        if (recover_prb):
            # Choose an adequate probe approximation
            prb = prb_init                        
        else:
            prb = prb_real.copy()
        result = slv.run_batch(
             data, psi, scan, prb, piter=piter, model=model, recover_prb=recover_prb)
        psi, prb = result['psi'], result['probe']

    # save result
    name = 'rec'+str(model)+str(nmodes)+'modes'+str(piter)+'iters'
    dxchange.write_tiff(np.angle(psi),
                        'rec/'+name+'/psiangle', overwrite=True)
    dxchange.write_tiff(np.abs(psi),  'rec/' +
                        name+'/psiamp', overwrite=True)
    dxchange.write_tiff_stack(np.angle(prb[0]),
                        'rec/'+name+'/prbangle', overwrite=True)
    dxchange.write_tiff_stack(np.abs(prb[0]), 'rec/' +
                        name+'/prbamp', overwrite=True)
