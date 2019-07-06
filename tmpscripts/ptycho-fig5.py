# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module for 3D ptychography."""

import dxchange
import tomopy
import xraylib as xl
import numpy as np
import scipy as sp
import pyfftw
import shutil
import warnings
warnings.filterwarnings("ignore")

PLANCK_CONSTANT = 6.58211928e-19  # [keV*s]
SPEED_OF_LIGHT = 299792458e+2  # [cm/s]


def wavelength(energy):
    """Calculates the wavelength [cm] given energy [keV].

    Parameters
    ----------
    energy : scalar

    Returns
    -------
    scalar
    """
    return 2 * np.pi * PLANCK_CONSTANT * SPEED_OF_LIGHT / energy


def wavenumber(energy):
    """Calculates the wavenumber [1/cm] given energy [keV].

    Parameters
    ----------
    energy : scalar

    Returns
    -------
    scalar
    """
    return 2 * np.pi / wavelength(energy)


class Material(object):
    """Material property definitions.

    Attributes
    ----------
    compound : string
        Molecular formula of the material.
    density : scalar
        Density of the compound [g/cm^3].
    energy : scalar
        Illumination energy [keV].
    """

    def __init__(self, compound, density, energy):
        self.compound = compound
        self.density = density
        self.energy = energy

    @property
    def beta(self):
        """Absorption coefficient."""
        return xl.Refractive_Index_Im(self.compound, self.energy, self.density)

    @property
    def delta(self):
        """Decrement of refractive index."""
        return 1 - xl.Refractive_Index_Re(self.compound, self.energy, self.density)


class Object(object):
    """Discrete object represented in a 3D regular grid.

    Attributes
    ----------
    beta : ndarray
        Absorption index.
    delta : ndarray
        Refractive index decrement.
    voxelsize : scalar [cm]
        Size of the voxels in the grid.
    """

    def __init__(self, beta, delta, voxelsize):
        self.beta = beta
        self.delta = delta
        self.voxelsize = voxelsize

    @property
    def shape(self):
        return self.beta.shape

    @property
    def complexform(self):
        return self.delta + 1j * self.beta


class Detector(object):
    """A planar area detector.

    Attributes
    ----------
    numx : int
        Number of horizontal pixels.
    numy : int
        Number of vertical pixels.
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y


def uniform(size):
    return np.ones((size, size), dtype='float32')


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


class Probe(object):
    """Illumination probe represented on a 2D regular grid.

    A finite-extent circular shaped probe is represented as 
    a complex wave. The intensity of the probe is maximum at 
    the center and damps to zero at the borders of the frame. 

    Attributes
    size : int
    ----------
        Size of the square 2D frame for the probe.
    rin : float
        Value between 0 and 1 determining where the 
        dampening of the intensity will start.
    rout : float
        Value between 0 and 1 determining where the 
        intensity will reach to zero.
    maxint : float
        Maximum intensity of the probe at the center.
    """

    def __init__(self, weights, maxint=1e5):
        self.weights = weights
        self.size = weights.shape[0]
        self.maxint = maxint
        self.shape = weights.shape

    @property
    def amplitude(self):
        """Amplitude of the probe wave"""
        return np.sqrt(self.maxint) * self.weights

    @property
    def phase(self):
        """Phase of the probe wave."""
        return 0.2 * self.weights

    @property
    def intensity(self):
        """Intensity of the probe wave."""
        return np.power(prb.amplitude, 2)

    @property
    def complex(self):
        return self.amplitude * np.exp(1j * self.phase)


class Scanner(object):
    def __init__(self, shape, sx, sy, margin=[0, 0], offset=[0, 0]):
        self.shape = shape
        self.sx = sx
        self.sy = sy
        self.margin = margin
        self.offset = offset

    @property
    def x(self):
        return np.arange(self.offset[0], self.shape[0]-self.margin[0]+1, self.sx)

    @property
    def y(self):
        return np.arange(self.offset[1], self.shape[1]-self.margin[1]+1, self.sy)


def scanner3(theta, shape, sx, sy, margin=[0, 0], offset=[0, 0], spiral=0):
    a = spiral
    scan = []
    for m in range(len(theta)):
        s = Scanner(shape, sx, sy, margin, offset=[
                    offset[0], np.mod(offset[1]+a, sy)])
        scan.append(s)
        a += spiral
    return scan


def _pad(phi, det):
    """Pads phi according to detector size."""
    npadx = (det.x - phi.shape[1]) // 2
    npady = (det.y - phi.shape[1]) // 2
    return np.pad(phi, ((0, 0), (npadx, npadx), (npady, npady)), mode='constant')


def project(obj, ang, energy):
    pb = tomopy.project(obj.beta, ang, pad=False) * obj.voxelsize
    pd = tomopy.project(obj.delta, ang, pad=False) * obj.voxelsize
    psi = np.exp(1j * wavenumber(energy) * (pd + 1j * pb))
    return psi


def exitwave(prb, psi, scan):
    return np.array([prb.complex * psi[i:i + prb.size, j:j + prb.size] for i in scan.x for j in scan.y], dtype='complex')


def propagate2(phi, det):
    phi = _pad(phi, det)
    intensity = np.abs(np.fft.fft2(phi)) ** 2
    return intensity.astype('float32')


def propagate3(prb, psi, scan, theta, det, noise=False):
    data = []
    for m in range(theta.size):
        phi = exitwave(prb, np.squeeze(psi[m]), scan[m])
        dat = propagate2(phi, det)
        if noise == True:
            dat = np.random.poisson(dat).astype('float32')
        data.append(dat)
    return data


def invptycho2(data, prb, scan, init, niter, rho, gamma, hobj, lamd, folder, debug):
    npadx = (data.shape[1] - prb.size) // 2
    npady = (data.shape[2] - prb.size) // 2
    psi = init.copy()
    convpsi = np.zeros(niter, dtype='float32')

    for i in range(niter):
        upd = np.zeros(psi.shape, dtype='complex')
        _psi = np.zeros(psi.shape, dtype='complex')
        a = 0
        for m in range(len(scan.x)):
            for n in range(len(scan.y)):

                # Near-plane.
                phi = np.multiply(
                    prb.complex, psi[scan.x[m]:scan.x[m] + prb.size, scan.y[n]:scan.y[n] + prb.size])

                # Go far-plane & Replace the amplitude with the measured amplitude.
                phi = np.pad(
                    phi, ((npadx, npadx), (npady, npady)), mode='constant')
                tmp = np.fft.fft2(phi)
                tmp = np.multiply(np.sqrt(data[a]), np.exp(1j * np.angle(tmp)))

                # Back to near-plane.
                iphi = np.fft.ifft2(tmp)

                # Object update.
                delphi = (iphi - phi)[npadx:npadx +
                                      prb.size, npady:npady+prb.size]
                num = np.multiply(np.conj(prb.complex), delphi)
                denum = np.power(np.abs(prb.complex), 2).max()
                _upd = np.true_divide(num, denum)

                upd[scan.x[m]:scan.x[m]+prb.size,
                    scan.y[n]:scan.y[n]+prb.size] += _upd
                a += 1

        # psi += upd.copy()
        # _psi = psi + (gamma / 2) * upd.copy()
        _psi = (1 - rho*gamma) * psi + rho*gamma * \
            (hobj - lamd/rho) + (gamma / 2) * upd.copy()

        convpsi[i] = np.linalg.norm(psi - _psi, ord='fro')
        psi = _psi.copy()

    return psi, convpsi


def invptycho3(data, prb, scan, init, theta, niter, rho, gamma, hobj, lamd, folder, debug):
    psi = np.zeros((init.shape), dtype='complex')
    convallpsi = np.zeros((theta.size, niter), dtype='float32')
    for m in range(theta.size):
        psi[m], convpsi = invptycho2(
            data[m], prb, scan[m], init[m], niter, rho, gamma, hobj[m], lamd[m], folder, debug)
        convallpsi[m] = convpsi
    return psi, convallpsi


def invtomo3(data, theta, voxelsize, energy, niter, init, eta):
    _data = 1 / wavenumber(energy) * np.log(data) / voxelsize
    pb = tomopy.recon(-np.real(_data), theta, algorithm='grad',
                      num_iter=niter, init_recon=init.beta.copy(), reg_par=eta)
    pd = tomopy.recon(np.imag(_data), theta, algorithm='grad',
                      num_iter=niter, init_recon=init.delta.copy(), reg_par=eta)
    obj = Object(pb, pd, 1e-6)
    return obj


folder = 'tmp/lego-joint-noise-1'
print(folder)

# Parameters.
rho = 0.5
gamma = 0.25
eta = 0.25
NITER = 256
piter = 1
titer = 1
maxint = 10
noise = False
debug = False

# Load a 3D object.
beta = dxchange.read_tiff(
    'data/test-beta-128.tiff').astype('float32')[::2, ::2, ::2]
delta = dxchange.read_tiff(
    'data/test-delta-128.tiff').astype('float32')[::2, ::2, ::2]
# # Load a 3D object.
# beta = dxchange.read_tiff('data/lego-imag.tiff')[::2, ::2, ::2]
# delta = dxchange.read_tiff('data/lego-real.tiff')[::2, ::2, ::2]

# Create object.
obj = Object(beta, delta, 1e-6)
dxchange.write_tiff(obj.beta, folder + '/beta')
dxchange.write_tiff(obj.delta, folder + '/delta')

# Create probe.
weights = gaussian(15, rin=0.8, rout=1.0)
prb = Probe(weights, maxint=maxint)
dxchange.write_tiff(prb.amplitude, folder + '/probe-amplitude')
dxchange.write_tiff(prb.phase, folder + '/probe-phase')

# Detector parameters.
det = Detector(63, 63)

# Define rotation angles.
theta = np.linspace(0, 2*np.pi, 180)

# Raster scan parameters for each rotation angle.
scan = scanner3(theta, beta.shape, 12, 12, margin=[
                prb.size, prb.size], offset=[0, 0], spiral=1)

# Project.
psis = project(obj, theta, energy=5)
dxchange.write_tiff(np.real(psis), folder + '/psi-amplitude')
dxchange.write_tiff(np.imag(psis), folder + '/psi-phase')

# Propagate.
data0 = propagate3(prb, psis, scan, theta, det, noise=False)
data = propagate3(prb, psis, scan, theta, det, noise=noise)
print(np.amax(data[3]))
print(np.amax(data[3]-data0[3]))
dxchange.write_tiff(np.fft.fftshift(
    np.log(np.array(data[0]))), folder + '/data')

# Init.
hobj = np.ones(psis.shape, dtype='complex')
psi = np.ones(psis.shape, dtype='complex')
lamd = np.zeros(psi.shape, dtype='complex')
tmp = np.zeros(obj.shape)
recobj = Object(tmp, tmp, 1e-6)

cp = np.zeros((NITER,))
cl = np.zeros((NITER,))
co = np.zeros((NITER,))

for m in range(NITER):

    # Ptychography.
    psi, conv = invptycho3(data, prb, scan, psi, theta, niter=piter, rho=rho,
                           gamma=gamma, hobj=hobj, lamd=lamd, folder=folder, debug=debug)
    dxchange.write_tiff(np.real(psi[0]).astype(
        'float32'), folder + '/psi-amplitude/psi-amplitude')
    dxchange.write_tiff(np.imag(psi[0]).astype(
        'float32'), folder + '/psi-phase/psi-phase')
    dxchange.write_tiff(np.abs(
        (psi + lamd/rho)[0]).astype('float32'), folder + '/psilamd-amplitude/psilamd-amplitude')
    dxchange.write_tiff(np.angle(
        (psi + lamd/rho)[0]).astype('float32'), folder + '/psilamd-phase/psilamd-phase')
    cp[m] = np.sqrt(np.sum(np.power(np.abs(hobj-psi), 2)))

    # Tomography.
    _recobj = invtomo3(psi + lamd/rho, theta, obj.voxelsize,
                       energy=5, niter=titer, init=recobj, eta=eta)
    co[m] = np.sqrt(
        np.sum(np.power(np.abs(recobj.complexform - _recobj.complexform), 2)))
    recobj = _recobj
    dxchange.write_tiff(
        recobj.beta[:, beta.shape[0] // 2], folder + '/beta/beta')
    dxchange.write_tiff(
        recobj.delta[:, delta.shape[0] // 2], folder + '/delta/delta')
    dxchange.write_tiff(recobj.beta, folder + '/beta-full/beta')
    dxchange.write_tiff(recobj.delta, folder + '/delta-full/delta')

    # Lambda update.
    hobj = project(recobj, theta, energy=5)
    dxchange.write_tiff(np.real(hobj[0]).astype(
        'float32'), folder + '/hobj-amplitude/hobj-amplitude')
    dxchange.write_tiff(np.imag(hobj[0]).astype(
        'float32'), folder + '/hobj-phase/hobj-phase')
    _lamd = lamd + 1 * rho * (psi - hobj)
    cl[m] = np.sqrt(np.sum(np.power(np.abs(lamd-_lamd), 2)))
    lamd = _lamd.copy()
    dxchange.write_tiff(np.abs(lamd[0]).astype(
        'float32'), folder + '/lamd-amplitude/lamd-amplitude')
    dxchange.write_tiff(np.angle(lamd[0]).astype(
        'float32'), folder + '/lamd-phase/lamd-phase')

    print(m, cp[m], co[m], cl[m])

    if np.isnan(cp[m]) or np.isnan(co[m]) or np.isnan(cl[m]):
        break

    np.save(folder + '/admm-conv-cp.npy', cp)
    np.save(folder + '/admm-conv-co.npy', co)
    np.save(folder + '/admm-conv-cl.npy', cl)
