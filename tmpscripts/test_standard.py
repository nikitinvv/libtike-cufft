import objects
import solver_gpu
import dxchange
# import tomopy
import numpy as np
import sys


def gaussian(size, rin=0.8, rout=1):
    r, c = np.mgrid[:size, :size] + 0.5
    rs = np.sqrt((r - size/2)**2 + (c - size/2)**2)
    rmax = np.sqrt(2) * 0.5 * rout * rs.max() + 1.0
    rmin = np.sqrt(2) * 0.5 * rin * rs.max()
    img = np.zeros((size, size), dtype='float32')
    img[rs < rmin] = 1.0
    img[rs > rmax] = 0.0
    zone = np.logical_and(rs > rmin, rs < rmax)
    img[zone] = np.divide(rmax - rs[zone], rmax - rmin)
    return img


def scanner3(theta, shape, sx, sy, margin=[0, 0], offset=[0, 0], spiral=0):
    a = spiral
    scan = []
    lenmaxx = 0
    lenmaxy = 0

    for m in range(len(theta)):
        s = objects.Scanner(shape, sx, sy, margin, offset=[
                            offset[0], np.mod(offset[1]+a, sy)])
        scan.append(s)
        a += spiral
        lenmaxx = max(lenmaxx, len(s.x))
        lenmaxy = max(lenmaxy, len(s.y))

    scanax = -1+np.zeros([len(theta), lenmaxx], dtype='int32')
    scanay = -1+np.zeros([len(theta), lenmaxy], dtype='int32')

    for m in range(len(theta)):
        scanax[m, :len(scan[m].x)] = scan[m].x
        scanay[m, :len(scan[m].y)] = scan[m].y

    return scan, scanax, scanay


if __name__ == "__main__":

    # Parameters fixed
    rho = 1e-10
    gamma = 0.25
    eta = 0.25/720/128/1e5*5
    piter = 512
    titer = 512
    NITER = 1
    voxelsize = 1e-6
    energy = 5

    # Load a 3D object.
    beta = dxchange.read_tiff(
        'data/test-beta-128.tiff').astype('float32')
    delta = dxchange.read_tiff(
        'data/test-delta-128.tiff').astype('float32')

    obj = objects.Object(beta, delta, voxelsize)
    det = objects.Detector(128, 128)
    theta = np.linspace(0, 2*np.pi, 720).astype('float32')
    tomoshape = [len(theta), obj.shape[0], obj.shape[2]]

    if(sys.argv[1] == "True"):
        print('compute regular')
        maxint = 4
        prb = objects.Probe(gaussian(16, rin=0.8, rout=1.0), maxint=maxint)
        scan, scanax, scanay = scanner3(theta, beta.shape, 6, 6, margin=[
                                        prb.size, prb.size], offset=[0, 0], spiral=1)
        slv = solver_gpu.Solver(prb, scanax, scanay,
                                theta, det, voxelsize, energy, tomoshape)
        # data
        psis = slv.fwd_tomo(obj.complexform)
        data = np.abs(slv.fwd_ptycho(slv.exptomo(psis)))**2

        # rec
        tau = 1e-16
        alpha = 1e-16
        h = np.ones(tomoshape, dtype='complex64', order='C')
        psi = np.ones(tomoshape, dtype='complex64', order='C')
        lamd = np.zeros(psi.shape, dtype='complex64', order='C')
        phi = np.zeros([3, *obj.shape], dtype='complex64', order='C')
        mu = np.zeros([3, *obj.shape], dtype='complex64', order='C')
        x = objects.Object(np.zeros(obj.shape, dtype='float32', order='C'), np.zeros(
            obj.shape, dtype='float32', order='C'), voxelsize)
        x = slv.admm(data, h, psi, phi, lamd, mu, x, rho, tau,
                     gamma, eta, alpha, piter, titer, NITER)
        dxchange.write_tiff(x.beta[64],  '../data_ptycho/beta/beta_st_60over6',overwrite=True)
        dxchange.write_tiff(x.delta[64],  '../data_ptycho/delta/delta_st_60over6',overwrite=True)

    # Denoise for different intensities
    maxinta = [4, 1, 0.1, 0.04]
    idc = np.int(sys.argv[2])
    print(idc)
    for k in range(0, 4):
        maxint = maxinta[k]
        prb = objects.Probe(gaussian(16, rin=0.8, rout=1.0), maxint=maxint)
        scan, scanax, scanay = scanner3(theta, beta.shape, 6, 6, margin=[
            prb.size, prb.size], offset=[0, 0], spiral=1)
        slv = solver_gpu.Solver(prb, scanax, scanay,
                                theta, det, voxelsize, energy, tomoshape)
        # data
        data = np.abs(slv.fwd_ptycho(
            slv.exptomo(slv.fwd_tomo(obj.complexform))))**2
        data *= (det.x*det.y)
        print(np.sqrt(np.amax(np.abs(data))))
        data = np.random.poisson(data).astype('float32')
        data /= (det.x*det.y)

        # rec
        tau = 1e-16
        alpha = 1e-16
        h = np.ones(tomoshape, dtype='complex64', order='C')
        psi = np.ones(tomoshape, dtype='complex64', order='C')
        lamd = np.zeros(psi.shape, dtype='complex64', order='C')
        phi = np.zeros([3, *obj.shape], dtype='complex64', order='C')
        mu = np.zeros([3, *obj.shape], dtype='complex64', order='C')
        x = objects.Object(np.zeros(obj.shape, dtype='float32', order='C'), np.zeros(
            obj.shape, dtype='float32', order='C'), voxelsize)
        x = slv.admm(data, h, psi, phi, lamd, mu, x, rho, tau,
                     gamma, eta, alpha, piter, titer, NITER)
        dxchange.write_tiff(
            x.beta[64],   '../data_ptycho/beta/beta_st_60over6_' + str(maxint)+'_maxint_noise', overwrite=True)
        dxchange.write_tiff(
            x.delta[64],  '../data_ptycho/delta/delta_st_60over6_'+str(maxint)+'_maxint_noise', overwrite=True)

        slv = []
