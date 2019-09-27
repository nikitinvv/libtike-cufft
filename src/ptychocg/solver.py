"""Module for 2D ptychography."""

import warnings
import numpy as np
import cupy as cp
import sys
import signal
from ptychocg.ptychofft import ptychofft


class Solver(object):
    def __init__(self, nscan, nprb, ndetx, ndety, ntheta, nz, n, ptheta):

        self.n = n  # object horizontal size
        self.nz = nz  # object vertical size
        self.ntheta = ntheta  # number of projections
        self.ptheta = ptheta  # number of projections for simultaneous processing on GPU
        self.nscan = nscan  # number of scan positions for 1 projection
        self.ndetx = ndetx  # detector x size
        self.ndety = ndety  # detector y size
        self.nprb = nprb  # probe size in 1 dimension

        # class for the ptycho transform (C++ wrapper)
        self.cl_ptycho = ptychofft(
            self.ptheta, self.nz, self.n, self.nscan, self.ndetx, self.ndety, self.nprb)
        # GPU memory deallocation with ctrl+C, ctrl+Z sygnals
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTSTP, self.signal_handler)

    # Free gpu memory after SIGINT, SIGSTSTP (destructor)
    def signal_handler(self, sig, frame):
        self = []
        sys.exit(0)

    # Ptychography transform (FQ)
    def fwd_ptycho(self, psi, scan, prb):
        res = cp.zeros([self.ptheta, self.nscan, self.ndety,
                        self.ndetx], dtype='complex64')
        self.cl_ptycho.fwd(res.data.ptr, psi.data.ptr,
                           scan.data.ptr, prb.data.ptr)  # C++ wrapper, send pointers to GPU arrays
        return res

    # Batch of Ptychography transform (FQ)
    def fwd_ptycho_batch(self, psi, scan, prb):
        data = np.zeros([self.ntheta, self.nscan, self.ndety,
                         self.ndetx], dtype='float32')
        for k in range(0, self.ntheta//self.ptheta):  # angle partitions in ptychography
            ids = np.arange(k*self.ptheta, (k+1)*self.ptheta)
            data0 = cp.abs(self.fwd_ptycho(
                psi[ids], scan[:, ids], prb[ids]))**2  # compute part on GPU
            data[ids] = data0.get()  # copy to CPU
        return data

    # Adjoint ptychography transform (Q*F*)
    def adj_ptycho(self, data, scan, prb):
        res = cp.zeros([self.ptheta, self.nz, self.n],
                       dtype='complex64')
        flg = 0  # compute adjoint operator with respect to object
        self.cl_ptycho.adj(res.data.ptr, data.data.ptr,
                           scan.data.ptr, prb.data.ptr, flg)  # C++ wrapper, send pointers to GPU arrays
        return res

    # Adjoint ptychography probe transform (O*F*), object is fixed
    def adj_ptycho_prb(self, data, scan, psi):
        res = cp.zeros([self.ptheta, self.nprb, self.nprb],
                       dtype='complex64')
        flg = 1  # compute adjoint operator with respect to probe
        self.cl_ptycho.adjq(res.data.ptr, data.data.ptr,
                            scan.data.ptr, psi.data.ptr, flg)  # C++ wrapper, send pointers to GPU arrays
        return res

    # Line search for the step sizes gamma
    def line_search(self, minf, gamma, u, fu, d, fd):
        while(minf(u, fu)-minf(u+gamma*d, fu+gamma*fd) < 0 and gamma > 1e-32):
            gamma *= 0.5
        if(gamma <= 1e-32):  # direction not found
            gamma = 0
            warnings.warn("Line search failed for conjugate gradient.")
        return gamma

    # Conjugate gradients for ptychography
    def cg_ptycho(self, data, psi, scan, prb, piter, model):
        assert prb.ndim == 3, "prb needs 3 dimensions, not %d" % prb.ndim
        # minimization functional
        def minf(psi, fpsi):
            if model == 'gaussian':
                f = cp.linalg.norm(cp.abs(fpsi)-cp.sqrt(data))**2
            elif model == 'poisson':
                f = cp.sum(cp.abs(fpsi)**2-2*data * cp.log(cp.abs(fpsi)+1e-32))
            return f

        # initial gradient steps
        gammapsi = 1 / (cp.max(cp.abs(prb)**2))
        assert cp.isfinite(gammapsi), "The probe amplitude cannot be zero."
        # gammaprb = 1 # # under development
        for i in range(piter):
            # 1) CG update psi with fixed prb
            fpsi = self.fwd_ptycho(psi, scan, prb)
            if model == 'gaussian':
                gradpsi = self.adj_ptycho(
                    fpsi-cp.sqrt(data)*cp.exp(1j*cp.angle(fpsi)), scan, prb)/(cp.max(cp.abs(prb)**2))
            elif model == 'poisson':
                gradpsi = self.adj_ptycho(
                    fpsi-data*fpsi/(cp.abs(fpsi)**2+1e-32), scan, prb)/(cp.max(cp.abs(prb)**2))
            # Dai-Yuan direction
            if i == 0:
                dpsi = -gradpsi
            else:
                dpsi = -gradpsi+cp.linalg.norm(gradpsi)**2 / \
                    ((cp.sum(cp.conj(dpsi)*(gradpsi-gradpsi0))))*dpsi
            gradpsi0 = gradpsi
            # line search
            fdpsi = self.fwd_ptycho(dpsi, scan, prb)
            gammapsi = self.line_search(minf, gammapsi, psi, fpsi, dpsi, fdpsi)
            # update psi
            psi = psi + gammapsi*dpsi

            # under development
            # # 2) CG update prb with fixed psi
            # fpsi = self.fwd_ptycho(psi, scan, prb)
            # if model == 'gaussian':
            #     gradprb = self.adj_ptychoq(
            #         fpsi-cp.sqrt(data)*cp.exp(1j*cp.angle(fpsi)), scan, psi)/self.nscan
            # elif model == 'poisson':
            #     gradprb = self.adj_ptychoq(
            #         fpsi-data*fpsi/(cp.abs(fpsi)**2+1e-32), scan, psi)/self.nscan
            # # Dai-Yuan direction
            # if i == 0:
            #     dprb = -gradprb
            # else:
            #     dprb = -gradprb+cp.linalg.norm(gradprb)**2 / \
            #         ((cp.sum(cp.conj(dprb)*(gradprb-gradprb0))))*dprb
            # gradprb0 = gradprb
            # # line search
            # fdprb = self.fwd_ptycho(psi, scan, dprb)
            # gammaprb = self.line_search(
            #     minf, gammaprb, psi, fpsi, psi, fdprb)
            # # update prb
            # prb = prb + gammaprb*dprb

            if (np.mod(i, 4) == 0):
                print("%d) gamma psi %.3e, residual %.3e" %
                      (i, gammapsi, minf(psi, fpsi)))

        if(cp.amax(cp.abs(cp.angle(psi))) > 3.14):
            print('possible phase wrap, max computed angle',
                  cp.amax(cp.abs(cp.angle(psi))))

        return psi, prb

    # Solve ptycho by angles partitions
    def cg_ptycho_batch(self, data, initpsi, scan, initprb, piter, model):
        assert prb.ndim == 3, "prb needs 3 dimensions, not %d" % prb.ndim
        psi = initpsi.copy()
        prb = initprb.copy()

        for k in range(0, self.ntheta//self.ptheta):  # angle partitions in ptychography
            ids = np.arange(k*self.ptheta, (k+1)*self.ptheta)
            datap = cp.array(data[ids])  # copy a part of data to GPU
            # solve cg ptychography problem for the part
            psi[ids], prb[ids] = self.cg_ptycho(
                datap, psi[ids], scan[:, ids], prb[ids, :, :], piter, model)
        return psi, prb
