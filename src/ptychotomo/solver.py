"""Module for 2D ptychography."""

import warnings

import cupy as cp
import numpy as np
import dxchange
from ptychotomo.ptychofft import ptychofft
import time

warnings.filterwarnings("ignore")


class Solver(object):
    def __init__(self, prb, scan, det, nz, n):

        self.scan = scan
        self.prb = prb
        self.nz = nz
        self.n = n
        self.nscan = scan.shape[2]
        self.ndety = det[0]
        self.ndetx = det[1]
        self.nprb = prb.shape[0]

        # create class for the ptycho transform
        self.cl_ptycho = ptychofft(
            1, self.nz, self.n, 1, self.nscan, self.ndety, self.ndetx, self.nprb)
        # normalization coefficients
        self.coefptycho = 1 / cp.abs(prb).max().get()
        self.coefdata = 1 / (self.ndetx*self.ndety *
                             (cp.abs(prb)**2).max().get())

    def mlog(self, psi):
        res = psi.copy()
        res[cp.abs(psi) < 1e-32] = 1e-32
        res = cp.log(res)
        return res

    # Ptychography transform (FQ)
    def fwd_ptycho(self, psi):
        res = cp.zeros([1, self.nscan, self.ndety,
                        self.ndetx], dtype='complex64', order='C')
        self.cl_ptycho.fwd(res.data.ptr, psi.data.ptr)
        res *= self.coefptycho  # normalization
        return res

    # Batch of Ptychography transform (FQ)
    def fwd_ptycho_batch(self, psi):
        data = np.zeros([1, self.nscan, self.ndety,
                         self.ndetx], dtype='float32')
        for k in range(0, 1):  # angle partitions in ptychography
            ids = np.arange(k, (k+1))
            self.cl_ptycho.setobj(
                self.scan[:, ids].data.ptr, self.prb.data.ptr)
            data0 = cp.abs(self.fwd_ptycho(psi[ids]))**2/self.coefdata
            data[ids] = data0.get()
        return data

    # Adjoint ptychography transform (Q*F*)
    def adj_ptycho(self, data):
        res = cp.zeros([1, self.nz, self.n],
                       dtype='complex64', order='C')
        self.cl_ptycho.adj(res.data.ptr, data.data.ptr)
        res *= self.coefptycho  # normalization
        return res

    # Line search for the step sizes gamma
    def line_search(self, minf, gamma, u, fu, d, fd):
        while(minf(u, fu)-minf(u+gamma*d, fu+gamma*fd) < 0 and gamma > 1e-12):
            gamma *= 0.5
        if(gamma <= 1e-12):  # direction not found
            #print('no direction')
            gamma = 0
        return gamma

    # Conjugate gradients for ptychography
    def cg_ptycho(self, data, init, piter, model):
        # minimization functional
        def minf(psi, fpsi):
            if model == 'gaussian':
                f = cp.linalg.norm(cp.abs(fpsi)-cp.sqrt(data))**2
            elif model == 'poisson':
                f = cp.sum(cp.abs(fpsi)**2-2*data * self.mlog(cp.abs(fpsi)))
            #f += rho*cp.linalg.norm(h-psi+lamd/rho)**2
            return f

        psi = init.copy()
        gamma = 2  # init gamma as a large value
        
        for i in range(piter):
            start = time.time()
            print(i)
            fpsi = self.fwd_ptycho(psi)
            if model == 'gaussian':
                grad = self.adj_ptycho(
                    fpsi-cp.sqrt(data)*cp.exp(1j*cp.angle(fpsi)))
            elif model == 'poisson':
                grad = self.adj_ptycho(fpsi-data*fpsi/(cp.abs(fpsi)**2+1e-32))
            #grad -= rho*(h - psi + lamd/rho)
            # Dai-Yuan direction
            if i == 0:
                d = -grad
            else:
                d = -grad+cp.linalg.norm(grad)**2 / \
                    ((cp.sum(cp.conj(d)*(grad-grad0))))*d
            grad0 = grad
            # line search
            fd = self.fwd_ptycho(d)
            gamma = self.line_search(minf, gamma, psi, fpsi, d, fd)
            psi = psi + gamma*d
            ##print(gamma,minf(psi, fpsi))
            # ang = cp.angle(psi)
            # ang[ang>np.pi]-=2*np.pi
            # ang[ang<-np.pi]+=2*np.pi
            # psi = cp.abs(psi)*cp.exp(1j*ang)
            end = time.time()
            print(end - start)
        if(cp.amax(cp.abs(cp.angle(psi))) > 3.14):
            print('possible phase wrap, max computed angle',
                  cp.amax(cp.abs(cp.angle(psi))))

        return psi

    # Solve ptycho by angles partitions
    def cg_ptycho_batch(self, data, init, piter, model):
        psi = init.copy()
        for k in range(0, 1//1):
            ids = np.arange(k*1, (k+1)*1)
            self.cl_ptycho.setobj(
                self.scan[:, ids].data.ptr, self.prb.data.ptr)
            datap = cp.array(data[ids])*self.coefdata  # normalized data
            psi[ids] = self.cg_ptycho(
                datap, psi[ids], piter, model)
        return psi
