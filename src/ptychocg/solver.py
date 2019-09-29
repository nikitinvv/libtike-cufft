"""Module for 2D ptychography."""

import warnings
import numpy as np
import cupy as cp
import sys
import signal
from ptychocg.ptychofft import ptychofft


class Solver(object):
    """Ptychographic solver instance.

    Manages GPU memory for solving the ptychography problem.

    Attribtues
    ----------
    nscan : int
        The number of scan positions at each angular view.
    nprb : int
        The pixel width and height of the probe illumination.
    ndetx, ndety : int
        The pixel width and height of the detector.
    ntheta : int
        The number of angular partitions of the data.
    n, nz : int
        The pixel width and height of the reconstructed grid.
    ptheta : int
        The number of angular partitions to process together
        simultaneously.

    """

    def __init__(self, nscan, nprb, ndetx, ndety, ntheta, nz, n, ptheta):
        """Please see help(Solver) for more info."""
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

    def signal_handler(self, sig, frame):
        """Free gpu memory after SIGINT, SIGSTSTP (destructor)"""
        self = []
        sys.exit(0)

    def fwd_ptycho(self, psi, scan, prb):
        """Ptychography transform (FQ)"""
        res = cp.zeros([self.ptheta, self.nscan, self.ndety,
                        self.ndetx], dtype='complex64')
        self.cl_ptycho.fwd(res.data.ptr, psi.data.ptr,
                           scan.data.ptr, prb.data.ptr)  # C++ wrapper, send pointers to GPU arrays
        return res

    def fwd_ptycho_batch(self, psi, scan, prb):
        """Batch of Ptychography transform (FQ)"""
        data = np.zeros([self.ntheta, self.nscan, self.ndety,
                         self.ndetx], dtype='float32')
        for k in range(0, self.ntheta//self.ptheta):  # angle partitions in ptychography
            ids = np.arange(k*self.ptheta, (k+1)*self.ptheta)
            data0 = cp.abs(self.fwd_ptycho(
                psi[ids], scan[:, ids], prb[ids]))**2  # compute part on GPU
            data[ids] = data0.get()  # copy to CPU
        return data

    def adj_ptycho(self, data, scan, prb):
        """Adjoint ptychography transform (Q*F*)"""
        res = cp.zeros([self.ptheta, self.nz, self.n],
                       dtype='complex64')
        flg = 0  # compute adjoint operator with respect to object
        self.cl_ptycho.adj(res.data.ptr, data.data.ptr,
                           scan.data.ptr, prb.data.ptr, flg)  # C++ wrapper, send pointers to GPU arrays
        return res

    def adj_ptycho_prb(self, data, scan, psi):
        """Adjoint ptychography probe transform (O*F*), object is fixed"""
        res = cp.zeros([self.ptheta, self.nprb, self.nprb],
                       dtype='complex64')
        flg = 1  # compute adjoint operator with respect to probe
        self.cl_ptycho.adj(psi.data.ptr, data.data.ptr,
                           scan.data.ptr, res.data.ptr, flg)  # C++ wrapper, send pointers to GPU arrays
        return res

    def line_search(self, minf, gamma, u, fu, d, fd):
        """Line search for the step sizes gamma"""
        while(minf(u, fu)-minf(u+gamma*d, fu+gamma*fd) < 0 and gamma > 1e-32):
            gamma *= 0.5
        if(gamma <= 1e-32):  # direction not found
            gamma = 0
            warnings.warn("Line search failed for conjugate gradient.")
        return gamma

    def cg_ptycho(self, data, psi, scan, prb, piter, piiter, model):
        """Conjugate gradients for ptychography"""
        assert prb.ndim == 3, "prb needs 3 dimensions, not %d" % prb.ndim
        # minimization functional

        def minf(psi, fpsi):
            if model == 'gaussian':
                f = cp.linalg.norm(cp.abs(fpsi)-cp.sqrt(data))**2
            elif model == 'poisson':
                f = cp.sum(cp.abs(fpsi)**2-2*data * cp.log(cp.abs(fpsi)+1e-32))
            return f

        # initial gradient steps       
        gammapsi = 1
        gammaprb = 1

        print("# congujate gradient parameters\n"
              "iteration, step size, function min")  # csv column headers
        for k in range(piter):          
                
            # object retrieval subproblem with fixed probe
            for i in range(piiter):
                # CG update psi with fixed prb
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
                gammapsi = self.line_search(
                    minf, gammapsi, psi, fpsi, dpsi, fdpsi)
                # update psi
                psi = psi + gammapsi*dpsi

            # probe retrieval subproblem with fixed object
            for i in range(piiter):
                fpsi = self.fwd_ptycho(psi, scan, prb)
                if model == 'gaussian':
                    gradprb = self.adj_ptycho_prb(
                        fpsi-cp.sqrt(data)*cp.exp(1j*cp.angle(fpsi)), scan, psi)/self.nscan

                elif model == 'poisson':
                    gradprb = self.adj_ptycho_prb(
                        fpsi-data*fpsi/(cp.abs(fpsi)**2+1e-32), scan, psi)/self.nscan
                # Dai-Yuan direction
                if (i == 0):
                    dprb = -gradprb
                else:
                    dprb = -gradprb+cp.linalg.norm(gradprb)**2 / \
                        ((cp.sum(cp.conj(dprb)*(gradprb-gradprb0))))*dprb
                gradprb0 = gradprb
                # line search
                fdprb = self.fwd_ptycho(psi, scan, dprb)
                gammaprb = self.line_search(
                    minf, gammaprb, psi, fpsi, psi, fdprb)
                # update prb
                prb = prb + gammaprb*dprb

            if (np.mod(k, 4) == 0):
                print("%4d, %.3e, %.3e, %.7e" %
                      (k, gammapsi, gammaprb, minf(psi, fpsi)))

        max_computed_angle = cp.amax(cp.abs(cp.angle(psi)))
        if max_computed_angle > 3.14:
            warnings.warn("Possible phase wrap. Max angle is %6f."
                          % max_computed_angle)

        return psi, prb

    def cg_ptycho_batch(self, data, initpsi, scan, initprb, piter, piiter, model):
        """Solve ptycho by angles partitions."""
        assert initprb.ndim == 3, "prb needs 3 dimensions, not %d" % initprb.ndim

        psi = initpsi.copy()
        prb = initprb.copy()

        for k in range(0, self.ntheta//self.ptheta):  # angle partitions in ptychography
            ids = np.arange(k*self.ptheta, (k+1)*self.ptheta)
            datap = cp.array(data[ids])  # copy a part of data to GPU
            # solve cg ptychography problem for the part
            psi[ids], prb[ids] = self.cg_ptycho(
                datap, psi[ids], scan[:, ids], prb[ids, :, :], piter, piiter, model)
        return psi, prb
