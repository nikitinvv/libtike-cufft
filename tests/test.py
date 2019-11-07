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
    nscan = 1000  # number of scan positions [max 5706 for the data example]
    nprb = 128  # probe size
    ndet = 128  # detector x size
    recover_prb = True  # True: recover probe, False: use the initial one
    # Reconstrucion parameters
    model = 'gaussian'  # minimization funcitonal (poisson,gaussian)
    piter = 128  # ptychography iterations
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
        # Compute intensity data on the detector |FQ|**2
        data = np.abs(slv.fwd_ptycho_batch(psi0, scan, prb0))**2
        dxchange.write_tiff(data, 'data', overwrite=True)

        # Initial guess
        psi = np.ones([ntheta, nz, n], dtype='complex64')
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
    dxchange.write_tiff(np.angle(psi),
                        'rec/psiang'+name, overwrite=True)
    dxchange.write_tiff(np.abs(psi),  'rec/prbamp'+name, overwrite=True)

    # recovered
    dxchange.write_tiff(np.angle(prb),
                        'rec/prbangle'+name, overwrite=True)
    dxchange.write_tiff(np.abs(prb),  'rec/prbamp'+name, overwrite=True)
    # init
    dxchange.write_tiff(np.angle(prb0),
                        'rec/prb0angle'+name, overwrite=True)
    dxchange.write_tiff(np.abs(prb0),
                        'rec/prb0amp'+name, overwrite=True)

    # plot result
    import matplotlib.pyplot as plt
    plt.figure(figsize=(11, 7))
    plt.subplot(2, 2, 1)
    plt.title('scan positions')
    plt.plot(scan[0, 0, :], scan[1, 0, :],
             '.', markersize=1.5, color='blue')
    plt.xlim([0, n])
    plt.ylim([0, nz])
    plt.gca().invert_yaxis()
    plt.subplot(2, 4, 1)
    plt.title('correct prb phase')
    plt.imshow(np.angle(prb0[0]), cmap='gray')
    plt.colorbar()
    plt.subplot(2, 4, 2)
    plt.title('correct prb amplitude')
    plt.imshow(np.abs(prb0[0]), cmap='gray')
    plt.colorbar()
    plt.subplot(2, 4, 3)
    plt.title('retrieved probe phase')
    plt.imshow(np.angle(prb[0]), cmap='gray')
    plt.colorbar()
    plt.subplot(2, 4, 4)
    plt.title('retrieved probe amplitude')
    plt.imshow(np.abs(prb[0]), cmap='gray')
    plt.colorbar()
    plt.subplot(2, 2, 3)
    plt.title('object phase')
    plt.imshow(np.angle(psi[0]), cmap='gray')
    plt.colorbar()
    plt.subplot(2, 2, 4)
    plt.title('object amplitude')
    plt.imshow(np.abs(psi[0]), cmap='gray')
    plt.colorbar()
    plt.savefig('result.png', dpi=600)
    print("See result.png and tiff files in rec/ folder")
