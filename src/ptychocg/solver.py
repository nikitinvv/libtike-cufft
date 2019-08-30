"""Module for 3D ptychography."""

import warnings

import cupy as cp
import numpy as np
import dxchange
import sys
import signal
from ptychocg.ptychofft import ptychofft


warnings.filterwarnings("ignore")

PLANCK_CONSTANT = 6.58211928e-19  # [keV*s]
SPEED_OF_LIGHT = 299792458e+2  # [cm/s]


class Solver(object):
    def __init__(self, prbmaxint, nscan, nprb, ndetx, ndety, ntheta, nz, n, ptheta):

        self.nscan = nscan
        self.nprb = nprb
        self.ntheta = ntheta
        self.nz = nz
        self.n = n
        self.nscan = nscan
        self.ndetx = ndetx
        self.ndety = ndety        
        self.nprb = nprb
        self.ptheta = ptheta
        
        # create class for the ptycho transform
        self.cl_ptycho = ptychofft(
            self.ptheta, self.nz, self.n, self.nscan, self.ndetx, self.ndety, self.nprb)
        # normalization coefficients
        self.coefptycho = 1 / np.sqrt(prbmaxint)
        # self.coefptychoq = 1 / np.sqrt(nscan)
        self.coefdata = 1 / (self.ndetx*self.ndety * prbmaxint)

        def signal_handler(sig, frame):  # Free gpu memory after SIGINT, SIGSTSTP
            slv = []
            sys.exit(0)
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTSTP, signal_handler)

    def mlog(self, psi):
        res = psi.copy()
        res[cp.abs(psi) < 1e-32] = 1e-32
        res = cp.log(res)
        return res

    # Ptychography transform (FQ)
    def fwd_ptycho(self, psi, scan, prb):
        res = cp.zeros([self.ptheta, self.nscan, self.ndety,
                        self.ndetx], dtype='complex64', order='C')
        self.cl_ptycho.fwd(res.data.ptr, psi.data.ptr,
                           scan.data.ptr, prb.data.ptr)
        res *= self.coefptycho  # normalization
        return res

    # Batch of Ptychography transform (FQ)
    def fwd_ptycho_batch(self, psi, scan, prb):
        data = np.zeros([self.ntheta, self.nscan, self.ndety,
                         self.ndetx], dtype='float32')
        for k in range(0, self.ntheta//self.ptheta):  # angle partitions in ptychography
            ids = np.arange(k*self.ptheta, (k+1)*self.ptheta)
            data0 = cp.abs(self.fwd_ptycho(
                psi[ids], scan[:, ids], prb[ids]))**2/self.coefdata
            data[ids] = data0.get()
        return data

    # Adjoint ptychography transform (Q*F*)
    def adj_ptycho(self, data, scan, prb):
        res = cp.zeros([self.ptheta, self.nz, self.n],
                       dtype='complex64', order='C')
        self.cl_ptycho.adj(res.data.ptr, data.data.ptr,
                           scan.data.ptr, prb.data.ptr)
        res *= self.coefptycho  # normalization
        return res

    # Adjoint ptychography transform (Q*F*)
    def adj_ptychoq(self, data, scan, psi):
        res = cp.zeros([self.ptheta, self.nprb, self.nprb],
                       dtype='complex64', order='C')
        self.cl_ptycho.adjq(res.data.ptr, data.data.ptr,
                           scan.data.ptr, psi.data.ptr)
        res *= self.coefptycho  # normalization
        return res        

    # Line search for the step sizes gamma
    def line_search(self, minf, gamma, u, fu, d, fd):
        while(minf(u, fu)-minf(u+gamma*d, fu+gamma*fd) < 0 and gamma > 1e-20):
            gamma *= 0.5
        if(gamma <= 1e-20):  # direction not found
            gamma = 0
        return gamma

    # Conjugate gradients for ptychography
    def cg_ptycho(self, data, init, scan, initq, piter, model):
        # minimization functional
        def minf(psi, fpsi):
            if model == 'gaussian':
                f = cp.linalg.norm(cp.abs(fpsi)-cp.sqrt(data))**2
            elif model == 'poisson':
                f = cp.sum(cp.abs(fpsi)**2-2*data * self.mlog(cp.abs(fpsi)))            
            return f

        psi = init.copy()        
        prb = initq.copy()        
        gamma = 0.03# init gamma as a large value
        gammaq = 1
        for i in range(piter):
            fpsi = self.fwd_ptycho(psi, scan, prb)
            if model == 'gaussian':                
                grad = self.adj_ptycho(fpsi-cp.sqrt(data)*cp.exp(1j*cp.angle(fpsi)), scan, prb)                                
            elif model == 'poisson':
                grad = self.adj_ptycho(
                    fpsi-data*fpsi/(cp.abs(fpsi)**2+1e-32), scan, prb)              
            # Dai-Yuan direction
            d = -grad                
            # if i == 0:
            #     d = -grad                
            # else:
            #     d = -grad+cp.linalg.norm(grad)**2 / \
            #         ((cp.sum(cp.conj(d)*(grad-grad0))))*d                
            # grad0 = grad
            # line search
            fd = self.fwd_ptycho(d, scan, prb)
            gamma = self.line_search(minf, gamma, psi, fpsi, d, fd)            
            psi = psi + gamma*d

            # update prb
            fpsi = self.fwd_ptycho(psi, scan, prb)
            if model == 'gaussian':
                tmp = fpsi-cp.sqrt(data)*cp.exp(1j*cp.angle(fpsi))
                gradq = self.adj_ptychoq(tmp, scan, psi)                                          
            elif model == 'poisson':
                grad = self.adj_ptycho(
                    fpsi-data*fpsi/(cp.abs(fpsi)**2+1e-32), scan, prb)              
            # # Dai-Yuan direction
            dq = -gradq
            # if i == 0:                
            #     dq = -gradq
            # else:                
            #     dq = -gradq+cp.linalg.norm(gradq)**2 / \
            #         ((cp.sum(cp.conj(dq)*(gradq-gradq0))))*dq
            # gradq0 = gradq
            # # line search
            # fdq = self.fwd_ptycho(psi, scan, dq)
            # gammaq = self.line_search(minf, gammaq, psi, fpsi, dq, fdq)            
            prb = prb + gammaq*dq
            
            print(gamma,cp.linalg.norm(gammaq*dq),cp.linalg.norm(gamma*d),minf(psi, fpsi))
            
        if(cp.amax(cp.abs(cp.angle(psi))) > 3.14):
            print('possible phase wrap, max computed angle',
                  cp.amax(cp.abs(cp.angle(psi))))

        return psi, prb

    # Solve ptycho by angles partitions
    def cg_ptycho_batch(self, data, init, scan, initq, piter, model):
        psi = init.copy()
        prb = initq.copy()
        for k in range(0, self.ntheta//self.ptheta):
            ids = np.arange(k*self.ptheta, (k+1)*self.ptheta)
            datap = cp.array(data[ids])*self.coefdata  # normalized data
            psi[ids],prb[ids] = self.cg_ptycho(
                datap, psi[ids], scan[:, ids], prb[ids], piter, model)
        return psi,prb

    
 