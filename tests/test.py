import numpy as np
import cupy as cp
import dxchange
from scipy import ndimage
import sys
import ptychocg as pt

if __name__ == "__main__":

    if (len(sys.argv) < 2):
        igpu = 0
    else:
        igpu = np.int(sys.argv[1])
    cp.cuda.Device(igpu).use()  # gpu id to use

    # set cupy to use unified memory
    pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
    cp.cuda.set_allocator(pool.malloc)

    # sizes
    n = 600  # horizontal size
    nz = 276  # vertical size
    ntheta = 1  # number of projections
    nscan = 1000  # number of scan positions [max 5706 for the data example]
    nprb = 128  # probe size
    ndetx = 128  # detector x size
    ndety = 128  # detector y size
    recover_prb = True  # True: recover probe, False: use the initial one
    # Reconstrucion parameters
    model = 'gaussian'  # minimization funcitonal (poisson,gaussian)
    piter = 128  # ptychography iterations
    ptheta = 1  # number of angular partitions for simultaneous processing in ptychography

    # read probe
    prb0 = cp.zeros([ntheta, nprb, nprb], dtype='complex64')
    prbamp = cp.array(dxchange.read_tiff(
        'model/prbamp.tiff').astype('float32'))
    prbang = cp.array(dxchange.read_tiff(
        'model/prbang.tiff').astype('float32'))
    prb0[0] = prbamp*cp.exp(1j*prbang)

    # read scan positions
    scan = cp.ones([2, ntheta, nscan], dtype='float32')
    scan[:, 0] = cp.load('model/coords.npy')[:, :nscan].astype('float32')

    # read object
    psi0 = cp.ones([ntheta, nz, n], dtype='complex64')
    psiamp = cp.array(dxchange.read_tiff(
        'model/initpsiamp.tiff').astype('float32'))
    psiang = cp.array(dxchange.read_tiff(
        'model/initpsiang.tiff').astype('float32'))
    psi0[0] = psiamp*cp.exp(1j*psiang)

    # Class gpu solver
    with pt.CGPtychoSolver(nscan, nprb, ndetx, ndety, ntheta, nz, n, ptheta) as slv:
        # Compute data
        data = slv.fwd_ptycho_batch(psi0, scan, prb0)
        dxchange.write_tiff(data, 'data', overwrite=True)

        # Initial guess
        psi = cp.ones([ntheta, nz, n], dtype='complex64')
        if (recover_prb):
            # Choose an adequate probe approximation
            prb = prb0.copy().swapaxes(1, 2)
        else:
            prb = prb0.copy()
        result = slv.run_batch(
            data, psi, scan, prb, piter=piter, model=model, recover_prb=recover_prb)
        psi, prb = result['psi'], result['prb']

    # Save result
    name = str(model)+str(piter)
    dxchange.write_tiff(cp.angle(psi).get(),
                        'rec/psiang'+name, overwrite=True)
    dxchange.write_tiff(cp.abs(psi).get(),  'rec/prbamp'+name, overwrite=True)

    # recovered
    dxchange.write_tiff(cp.angle(prb).get(),
                        'rec/prbangle'+name, overwrite=True)
    dxchange.write_tiff(cp.abs(prb).get(),  'rec/prbamp'+name, overwrite=True)
    # init
    dxchange.write_tiff(cp.angle(prb0).get(),
                        'rec/prb0angle'+name, overwrite=True)
    dxchange.write_tiff(cp.abs(prb0).get(),
                        'rec/prb0amp'+name, overwrite=True)

    # plot result
    import matplotlib.pyplot as plt
    plt.figure(figsize=(11, 7))
    plt.subplot(2, 2, 1)
    plt.title('scan positions')
    plt.plot(scan[0, 0, :].get(), scan[1, 0, :].get(),
             '.', markersize=1.5, color='blue')
    plt.xlim([0, n])
    plt.ylim([0, nz])
    plt.gca().invert_yaxis()
    plt.subplot(2, 4, 1)
    plt.title('correct prb phase')
    plt.imshow(cp.angle(prb0[0]).get(), cmap='gray')
    plt.colorbar()
    plt.subplot(2, 4, 2)
    plt.title('correct prb amplitude')
    plt.imshow(cp.abs(prb0[0]).get(), cmap='gray')
    plt.colorbar()
    plt.subplot(2, 4, 3)
    plt.title('retrieved probe phase')
    plt.imshow(cp.angle(prb[0]).get(), cmap='gray')
    plt.colorbar()
    plt.subplot(2, 4, 4)
    plt.title('retrieved probe amplitude')
    plt.imshow(cp.abs(prb[0]).get(), cmap='gray')
    plt.colorbar()
    plt.subplot(2, 2, 3)
    plt.title('object phase')
    plt.imshow(cp.angle(psi[0]).get(), cmap='gray')
    plt.colorbar()
    plt.subplot(2, 2, 4)
    plt.title('object amplitude')
    plt.imshow(cp.abs(psi[0]).get(), cmap='gray')
    plt.colorbar()
    plt.savefig('result.png', dpi=600)
    print("See result.png and tiff files in rec/ folder")
