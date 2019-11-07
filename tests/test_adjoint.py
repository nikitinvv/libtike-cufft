import numpy as np
import dxchange
from scipy import ndimage
import sys
import ptychocg as pt

if __name__ == "__main__":

    if (len(sys.argv) < 2):
        igpu = 0
    else:
        igpu = np.int(sys.argv[1])

    # sizes
    n = 600  # horizontal size
    nz = 276  # vertical size
    ntheta = 1  # number of projections
    nscan = 100  # number of scan positions [max 5706 for the data example]
    nprb = 128  # probe size
    ndet = 128  # detector x size
    ptheta = 1  # number of angular partitions for simultaneous processing in ptychography

    # read probe
    prb0 = np.zeros([ntheta, nprb, nprb], dtype='complex64')
    prbamp = dxchange.read_tiff('model/prbamp.tiff').astype('float32')
    prbang = dxchange.read_tiff('model/prbang.tiff').astype('float32')
    prb0[0] = prbamp*np.exp(1j*prbang)

    # read scan positions
    scan = np.ones([ntheta, nscan, 2], dtype='float32')
    scan[0] = np.moveaxis(np.load('model/coords.npy'), 0, 1)[:nscan]

    # read object
    psi0 = np.ones([ntheta, nz, n], dtype='complex64')
    psiamp = dxchange.read_tiff('model/initpsiamp.tiff').astype('float32')
    psiang = dxchange.read_tiff('model/initpsiang.tiff').astype('float32')
    psi0[0] = psiamp*np.exp(1j*psiang)

    # Class gpu solver
    with pt.CGPtychoSolver(nscan, nprb, ndet, ntheta, nz, n, ptheta, igpu) as slv:
        # Compute forward operator FQpsi
        t1 = slv.fwd_ptycho_batch(psi0, scan, prb0)
        t2 = slv.adj_ptycho_batch(t1, scan, prb0)
        t3 = slv.adj_ptycho_batch_prb(t1, scan, psi0)
        a = np.sum(psi0*np.conj(t2))
        b = np.sum(t1*np.conj(t1))
        c = np.sum(prb0*np.conj(t3))
        print('Adjoint test')
        print('<FQP,FQP> = ', a)
        print('<P,Q*F*FQP> = ', b)
        print('<Q,P*F*FPQ> = ', c)
        print('<FQP,FQP> - <P,Q*F*FQP> = ', a-b)         
        print('<FQP,FQP> - <Q,P*F*FPQ> = ', a-c)         
        if (((a-b)/a<1e-3)&((a-c)/a<1e-3)):
            print('PASSED')
        else:
            print('NOT PASSED')            