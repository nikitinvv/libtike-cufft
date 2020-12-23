import sys

import dxchange
import cupy as cp
import numpy as np
from scipy import ndimage

import libtike.cufft as pt
from scipy.ndimage import zoom

# 3d scanner positions
def scanner3(theta, shape, sx, sy, psize, spiral=0, randscan=False, save=False):
    scx, scy = np.meshgrid(
        np.arange(0, shape[1]-psize+1, sx), np.arange(0, shape[0]-psize+1, sy))
    shapescan = np.size(scx)
    scanax = -1+np.zeros([len(theta), shapescan], dtype='float32')
    scanay = -1+np.zeros([len(theta), shapescan], dtype='float32')
    a = spiral
    for m in range(len(theta)):
        scanax[m] = np.ndarray.flatten(scx)+np.mod(a, sx)
        scanay[m] = np.ndarray.flatten(scy)
        a += spiral
        if randscan:
            #scanax[m] += sx*(np.random.random(1)-0.5)*1
            #scanay[m] += sy*(np.random.random(1)-0.5)*1
            scanax[m] += sx*(np.random.random(shapescan)-0.5)*0.5
            scanay[m] += sy*(np.random.random(shapescan)-0.5)*0.5
            # print(scanax[m])        
    scanax[np.where(np.round(scanax) < 0)] = 0
    scanay[np.where(np.round(scanay) < 0)] = 0
    scanax[np.where(np.round(scanax) > shape[1]-psize)] = shape[1]-psize-1
    scanay[np.where(np.round(scanay) > shape[0]-psize)] = shape[0]-psize-1
    # plot probes
    if save:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        def random_color():
            rgbl=[0.75,0.75,0]
            #np.random.shuffle(rgbl)
            return tuple(rgbl)
        for j in range(0,1):
            fig, ax = plt.subplots(1)
            plt.xlim(-1, shape[1]+2)
            plt.ylim(-1, shape[0]+2)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.axis('off')
            for k in range(0, len(scanax[j])):
                if(scanax[j, k] < 0 or scanay[j, k] < 0):
                    continue
                c = patches.Circle(
                    (scanax[j, k]+psize//2, scanay[j, k]+psize//2), psize//2, fill=False, edgecolor=[*random_color(),1], linewidth=8)
                ax.add_patch(c)
            
            plt.savefig('scan'+str(j)+'.png')
            
    scan = np.zeros([2,len(theta), shapescan], dtype='float32',order='C')             
    scan[0]=scanax
    scan[1]=scanay
    return scan
if __name__ == "__main__":
    
    
    c = float(sys.argv[1])
    st = int(sys.argv[2])
    mask_flg = int(sys.argv[3])
    # sizes
    n = 512  # horizontal size
    nz = 512  # vertical size
    ntheta = 1  # number of projections
    nprb = 128  # probe size
    ndet = 128  # detector x size
    recover_prb = True  # True: recover probe, False: use the initial one
    # Reconstrucion parameters
    model = 'gaussian'  # minimization funcitonal (poisson,gaussian)
    piter = 512  # ptychography iterations
    
    ptheta = 1  # number of angular partitions for simultaneous processing in ptychography
    nmodes = 1 # number of probe modes for decomposition

    # read probe
    psi0 = np.zeros([ntheta,nz,n],dtype='complex64')
    psiamp = dxchange.read_tiff('model/initpsiamp.tiff').astype('float32')
    psiang = dxchange.read_tiff('model/initpsiang.tiff').astype('float32')
    psi0[0]=zoom(psiamp[5:-15,285:541],2,order=2)*np.exp(1j*zoom(psiang[5:-15,285:541],2,order=2))

    prb0 = np.zeros([ntheta,nmodes, nprb, nprb], dtype='complex64')
    prbamp = dxchange.read_tiff('model/prbamp.tiff').astype('float32')*c
    prbang = dxchange.read_tiff('model/prbang.tiff').astype('float32')
    prb0[0,0] = prbamp*np.exp(1j*prbang)

    # read scan positions    
    scan0 = scanner3([0], [nz-1,n-1], 8, 8, nprb, spiral=0, randscan=True, save=True).swapaxes(0,1).swapaxes(1,2).astype('float32')    
    scan = scan0.copy()
    nscan = scan.shape[1]
    #nscan=4
    scan[0, :nscan, 0] = scan0[0,:nscan, 1]
    scan[0, :nscan, 1] = scan0[0,:nscan, 0]
    
    # mask = np.zeros([ndet,ndet],dtype='float32')
    # mask[32:-32,32:-32]=1
    # Class gpu solver
    with pt.CGPtychoSolver(nscan, nprb, ndet, ptheta, nz, n) as slv:
        # Compute intensity data on the detector |FQ|**2
        data = np.abs(slv.fwd_ptycho_batch(psi0, scan, prb0))**2        
        print(data.max())
        data = np.random.poisson(data).astype('float32')
        # dxchange.write_tiff(np.fft.fftshift(data), 'data0', overwrite=True)
        # if(mask_flg==1):
        #     data = np.fft.fftshift(np.fft.fftshift(data)*mask)
        #     prb*=mask

        # data = data*mask
        # prb0 = prb0*mask
        #dxchange.write_tiff(np.fft.fftshift(data), 'data', overwrite=True)
    if (mask_flg==1):
        prb0 = prb0[:,:,8:-8,8:-8]
        data = np.fft.fftshift(np.fft.fftshift(data,axes=(2,3))[:,:,8:-8,8:-8],axes=(2,3))
        nprb = prb0.shape[-1]
        ndet = data.shape[-1]
        scan+=8
    name = str(model)+str(piter)+'half'+str(c)+'_'+str(mask_flg)+'_'+str(st)
    dxchange.write_tiff(data,
                        'data/data'+name, overwrite=True)
    with pt.CGPtychoSolver(nscan, nprb, ndet, ptheta, nz, n) as slv:
        print(prb0.shape)
        print(data.shape)
        # Initial guess
        psi = np.ones([ntheta, nz, n], dtype='complex64')
        prb = prb0.copy()
        result = slv.run_batch(
            data, psi, scan, prb, piter=piter, model=model, recover_prb=recover_prb)
        psi, prb = result['psi'], result['probe']

    # Save result
    
    dxchange.write_tiff(np.angle(psi),
                        'rec/psiang'+name, overwrite=True)
    