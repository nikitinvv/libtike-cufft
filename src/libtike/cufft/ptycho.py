"""A module for ptychography solvers.

This module implements ptychographic solvers which all inherit from a
ptychography base class. The base class implements the forward and adjoint
ptychography operators and manages GPU memory.

Solvers in this module are Python context managers which means they should be
instantiated using a with-block. e.g.

```python
# load data and such
data = cp.load(...)
# instantiate the solver with memory allocation related parameters
with CustomPtychoSolver(...) as solver:
    # call the solver with solver specific parameters
    result = solver.run(data, ...)
# solver memory is automatically freed at with-block exit
```

Context managers are capable of gracefully handling interruptions (CTRL+C).

"""

import signal
import sys
import warnings

import cupy as cp
import numpy as np
import dxchange
from libtike.cufft.ptychofft import ptychofft
from skimage.feature import register_translation

class PtychoCuFFT(ptychofft):
    """Base class for ptychography solvers using the cuFFT library.

    This class is a context manager which provides the basic operators required
    to implement a ptychography solver. It also manages memory automatically,
    and provides correct cleanup for interruptions or terminations.

    Attribtues
    ----------
    nscan : int
        The number of scan positions at each angular view.
    nprb : int
        The pixel width and height of the probe illumination.
    ndet, ndet : int
        The pixel width and height of the detector.
    ntheta : int
        The number of angular partitions of the data.
    n, nz : int
        The pixel width and height of the reconstructed grid.
    """

    array_module = cp
    asnumpy = cp.asnumpy

    def __init__(self, nscan, probe_shape, detector_shape, ntheta, nz, n):
        """Please see help(PtychoCuFFT) for more info."""
        super().__init__(ntheta, nz, n, nscan, detector_shape, probe_shape)

    def __enter__(self):
        """Return self at start of a with-block."""
        return self

    def __exit__(self, type, value, traceback):
        """Free GPU memory due at interruptions or with-block exit."""
        self.free()

    @classmethod
    def _batch(self, function, output, *inputs):
        """Does data shuffle between host and device."""
        xp = self.array_module
        # TODO: handle the case when ptheta does not divide ntheta evenly
        for ids in range(0, inputs[0].shape[0]):
            inputs_gpu = [xp.array(x[ids:ids+1]) for x in inputs]
            output[ids] = function(*inputs_gpu).get()
        return output

    def fwd(self, psi, scan, probe):
        """Ptychography transform (FQ)."""
        assert psi.dtype == cp.complex64, f"{psi.dtype}"
        assert scan.dtype == cp.float32, f"{scan.dtype}"
        assert probe.dtype == cp.complex64, f"{probe.dtype}"
        farplane = cp.zeros([self.ptheta, self.nscan, self.ndet, self.ndet],
                            dtype='complex64')
        ptychofft.fwd(self, farplane.data.ptr, psi.data.ptr,
                      scan.data.ptr, probe.data.ptr)
        return farplane
        
    def fwd_ptycho_batch(self, psi, scan, probe):
        """Batch of Ptychography transform (FQ)."""
        data = np.zeros([scan.shape[0], self.nscan, self.ndet, self.ndet],
                        dtype='complex64')
        return self._batch(self.fwd, data, psi, scan, probe)

    def adj(self, farplane, scan, probe):
        """Adjoint ptychography transform (Q*F*)."""
        assert farplane.dtype == cp.complex64, f"{farplane.dtype}"
        assert scan.dtype == cp.float32, f"{scan.dtype}"
        assert probe.dtype == cp.complex64, f"{probe.dtype}"
        psi = cp.zeros([self.ptheta, self.nz, self.n], dtype='complex64')
        flg = 0  # compute adjoint operator with respect to object
        ptychofft.adj(self, psi.data.ptr, farplane.data.ptr,
                      scan.data.ptr, probe.data.ptr, flg)
        return psi

    def adj_ptycho_batch(self, farplane, scan, probe):
        """Batch of Ptychography transform (FQ)."""
        psi = np.zeros([scan.shape[0], self.nz, self.n], dtype='complex64')
        return self._batch(self.adj, psi, farplane, scan, probe)

    def adj_probe(self, farplane, scan, psi):
        """Adjoint ptychography probe transform (O*F*), object is fixed."""
        assert farplane.dtype == cp.complex64, f"{farplane.dtype}"
        assert scan.dtype == cp.float32, f"{scan.dtype}"
        assert psi.dtype == cp.complex64, f"{psi.dtype}"
        probe = cp.zeros([self.ptheta, self.nprb, self.nprb],
                         dtype='complex64')
        flg = 1  # compute adjoint operator with respect to probe
        ptychofft.adj(self, psi.data.ptr, farplane.data.ptr,
                      scan.data.ptr, probe.data.ptr, flg)
        return probe

    def adj_ptycho_batch_prb(self, farplane, scan, psi):
        """Batch of Ptychography transform (FQ)."""
        probe = np.zeros(
            [scan.shape[0], self.nprb, self.nprb], dtype='complex64')
        return self._batch(self.adj_probe, probe, farplane, scan, psi)

    def run(self, data, psi, scan, probe, **kwargs):
        """Placehold for a child's solving function."""
        raise NotImplementedError("Cannot run a base class.")

    def run_batch(self, data, psi, scan, probe, **kwargs):
        """Run by dividing the work into batches."""
        assert probe.ndim == 4, "probe needs 4 dimensions, not %d" % probe.ndim

        psi = psi.copy()
        probe = probe.copy()

        # angle partitions in ptychography
        for k in range(0, scan.shape[0] // self.ptheta):
            ids = np.arange(k * self.ptheta, (k + 1) * self.ptheta)
            # copy to GPU
            psi_gpu = cp.array(psi[ids])
            scan_gpu = cp.array(scan[ids])
            prb_gpu = cp.array(probe[ids])
            data_gpu = cp.array(data[ids])
            # solve cg ptychography problem for the part
            result = self.run(
                data_gpu,
                psi_gpu,
                scan_gpu,
                prb_gpu,
                **kwargs,
            )
            psi[ids], probe[ids] = result['psi'].get(), result['probe'].get()
        return {
            'psi': psi,
            'probe': probe,
        }
def _upsampled_dft_batch(data, ups,
                   upsample_factor=1, axis_offsets=None):
   
    im2pi = 1j * 2 * np.pi
    # rec1 = np.zeros([data.shape[0],ups.astype(int),ups.astype(int)],dtype='complex128')
    # for k in range(data.shape[0]):
    #     tdata = data[k]
    #     dim_properties = list(zip(tdata.shape, axis_offsets[k]))
    #    # print(dim_properties)        
    #     for (n_items, ax_offset) in dim_properties[::-1]:
    #         kernel = ((np.arange(ups) - ax_offset)[:, None]
    #                   * np.fft.fftfreq(n_items, upsample_factor))
    #         kernel = np.exp(-im2pi * kernel)
    #         tdata = np.tensordot(kernel, tdata, axes=(1, -1))            
    #     rec1[k] = tdata
    
    tdata = data.copy()
    kernel = (cp.tile(cp.arange(ups),(data.shape[0],1))-axis_offsets[:,1:2])[:,:,None]*cp.fft.fftfreq(data.shape[2], upsample_factor)
    kernel = cp.exp(-im2pi * kernel)
    tdata = cp.einsum('ijk,ipk->ijp',kernel,tdata)
    kernel = (cp.tile(cp.arange(ups),(data.shape[0],1))-axis_offsets[:,0:1])[:,:,None]*cp.fft.fftfreq(data.shape[2], upsample_factor)
    kernel = cp.exp(-im2pi * kernel)
    rec = cp.einsum('ijk,ipk->ijp',kernel,tdata)
    
    
    return rec

def register_translation_batch(src_image, target_image, upsample_factor=1,
                         space="real"):
    # assume complex data is already in Fourier space
    if space.lower() == 'fourier':
        src_freq = src_image
        target_freq = target_image
    # real data needs to be fft'd.
    elif space.lower() == 'real':
        src_freq = cp.fft.fft2(src_image)
        target_freq = cp.fft.fft2(target_image)
    
    # Whole-pixel shift - Compute cross-correlation by an IFFT
    shape = src_freq.shape
    image_product = src_freq * target_freq.conj()
    cross_correlation = cp.fft.ifft2(image_product)
    A = cp.abs(cross_correlation)                          
    maxima = A.reshape(A.shape[0],-1).argmax(1)
    maxima = cp.column_stack(cp.unravel_index(maxima,A[0,:,:].shape))

    midpoints = np.array([cp.fix(axis_size / 2) for axis_size in shape[1:]])

    shifts = cp.array(maxima, dtype=cp.float64)
    ids = cp.where(shifts[:,0] > midpoints[0])
    shifts[ids[0],0] -= shape[1]
    ids = cp.where(shifts[:,1] > midpoints[1])
    shifts[ids[0],1] -= shape[2]
    if upsample_factor > 1:
        # Initial shift estimate in upsampled grid
        shifts = np.round(shifts * upsample_factor) / upsample_factor
        upsampled_region_size = np.ceil(upsample_factor * 1.5)
        # Center of output array at dftshift + 1
        dftshift = np.fix(upsampled_region_size / 2.0)
        
        normalization = (src_freq[0].size * upsample_factor ** 2)
        # Matrix multiply DFT around the current shift estimate
     
        sample_region_offset = dftshift - shifts*upsample_factor
        cross_correlation = _upsampled_dft_batch(image_product.conj(),
                                           upsampled_region_size,
                                           upsample_factor,
                                           sample_region_offset).conj()
        cross_correlation /= normalization
        # Locate maximum and map back to original pixel grid
        A = cp.abs(cross_correlation)                          
        maxima = A.reshape(A.shape[0],-1).argmax(1)
        maxima = cp.column_stack(cp.unravel_index(maxima,A[0,:,:].shape))

        maxima = cp.array(maxima, dtype=cp.float64) - dftshift

        shifts = shifts + maxima / upsample_factor       

    # If its only one row or column the shift along that dimension has no
    # effect. We set to zero.
    for dim in range(src_freq.ndim):
        if shape[dim] == 1:
            shifts[dim] = 0

    
    return shifts

class CGPtychoSolver(PtychoCuFFT):
    """Solve the ptychography problem using congujate gradient."""

    @staticmethod
    def line_search_sqr(f, p1, p2, p3, step_length=1, step_shrink=0.5):
        """Optimized line search for square functions
            Example of otimized computation for the Gaussian model:
            sum_j|G_j(psi+gamma dpsi)|^2 = sum_j|G_j(psi)|^2+
                                           gamma^2*sum_j|G_j(dpsi)|^2+
                                           gamma*sum_j (G_j(psi).real*G_j(psi).real+2*G_j(dpsi).imag*G_j(dpsi).imag)
            p1,p2,p3 are temp variables to avoid computing the fwd operator during the line serch
            p1 = sum_j|G_j(psi)|^2
            p2 = sum_j|G_j(dpsi)|^2
            p3 = sum_j (G_j(psi).real*G_j(psi).real+2*G_j(dpsi).imag*G_j(dpsi).imag)

            Parameters	
            ----------	
            f : function(x)	
                The function being optimized.	
            p1,p2,p3 : vectors	
                Temporarily vectors to avoid computing forward operators        
        """
        assert step_shrink > 0 and step_shrink < 1
        m = 0  # Some tuning parameter for termination
        fp1 = f(p1) # optimize computation
        # Decrease the step length while the step increases the cost function
        while f(p1+step_length**2 * p2+step_length*p3) > fp1 + step_shrink * m:
            if step_length < 1e-32:
                warnings.warn("Line search failed for conjugate gradient.")
                return 0
            step_length *= step_shrink            
        return step_length
    
    def run(
            self,
            data,
            psi,
            scan,
            probe,
            piter,
            model='gaussian',
            recover_prb=False,
            ortho_prb=False,
    ):
        """Conjugate gradients for ptychography.

        Parameters
        ----------
        model : str gaussian or poisson
            The noise model to use for the gradient.
        piter : int
            The number of gradient steps to take.
        recover_prb : bool
            Whether to recover the probe or assume the given probe is correct.

        """
        assert probe.ndim == 4, "probe needs 4 dimensions, not %d" % probe.ndim
        # minimization functional
        def minf(fpsi):
            if model == 'gaussian':
                f = cp.linalg.norm(cp.sqrt(cp.abs(fpsi)) - cp.sqrt(data))**2
            elif model == 'poisson':
                f = cp.sum(
                    cp.abs(fpsi) - data * cp.log(cp.abs(fpsi) + 1e-32))
            return f
        dprb=0
        dpsi=0
        gradprb0=0
        gradpsi0=0
        print("# congujate gradient parameters\n"
              "iteration, step size object, step size probe, function min"
              )  # csv column headers
        gammaprb = 0
        
        #dslv = dc.SolverDeform(self.nscan, self.nprb, self.nprb)
        for i in range(piter):
            
            # 1) object retrieval subproblem with fixed probes
            # sum of forward operators associated with each probe
            # sum of abs value of forward operators
            absfpsi = data*0
            for k in range(probe.shape[1]):
                tmp = self.fwd(psi, scan, probe[:, k])
                absfpsi += np.abs(tmp)**2
            
            # a=cp.linalg.norm(cp.sqrt(absfpsi[0,:]))
            # b=cp.linalg.norm(cp.sqrt(data[0,:]))
            # probe*=(b/a)
            # absfpsi*=(b/a)**(1+(i==0))
            # gradprb0*=(b/a)
            # dprb*=(b/a)
                        
            a = cp.sum(cp.sqrt(absfpsi*data))
            b = cp.sum(absfpsi)
            probe *= (a/b)
            absfpsi *= (a/b)**2
            # take gradients
            gradpsi = cp.zeros(
                    [self.ptheta, self.nz, self.n], dtype='complex64')
            if model == 'gaussian':                
                for k in range(probe.shape[1]):
                    fpsi = self.fwd(psi, scan, probe[:, k])*(b/a)                
                    gradpsi += self.adj(
                        fpsi - cp.sqrt(data) * fpsi/(cp.sqrt(absfpsi)+1e-32),
                        scan,
                        probe[:, k],
                    ) / (cp.max(cp.abs(probe[:, k]))**2)
            elif model == 'poisson':
                for k in range(probe.shape[1]):
                    gradpsi += self.adj(
                        fpsi - data * fpsi / (absfpsi + 1e-32),
                        scan,
                        probe[:, k],
                    ) / (cp.max(cp.abs(probe[:, k]))**2)
            # Dai-Yuan direction
            #dpsi = -gradpsi
            if i == 0:
                dpsi = -gradpsi
            else:
                dpsi = -gradpsi + (
                    cp.linalg.norm(gradpsi)**2 /
                    (cp.sum(cp.conj(dpsi) * (gradpsi - gradpsi0))) * dpsi)
            gradpsi0 = gradpsi
            
	        
            # Use optimized line search for square functions, note:
            # sum_j|G_j(psi+gamma dpsi)|^2 = sum_j|G_j(psi)|^2+
            #                               gamma^2*sum_j|G_j(dpsi)|^2+
            #                               gamma*sum_j (G_j(psi).real*G_j(psi).real+2*G_j(dpsi).imag*G_j(dpsi).imag)
            # temp variables to avoid computing the fwd operator during the line serch
            #p1 = sum_j|G_j(psi)|^2
            #p2 = sum_j|G_j(dpsi)|^2
            #p3 = sum_j (G_j(psi).real*G_j(psi).real+2*G_j(dpsi).imag*G_j(dpsi).imag)
            p1 = data*0 
            p2 = data*0
            p3 = data*0
            for k in range(probe.shape[1]):
                tmp1 = self.fwd(psi, scan, probe[:, k])
                tmp2 = self.fwd(dpsi, scan, probe[:, k])
                p1 += cp.abs(tmp1)**2 
                p2 += cp.abs(tmp2)**2
                p3 += 2*(tmp1.real*tmp2.real+tmp1.imag*tmp2.imag)
            # line search		
            gammapsi = 0.5*self.line_search_sqr(minf,p1,p2,p3)
            

            # position correction
            
            if(i>0):
                tmp1 = self.fwd(psi, scan, probe[:, 0]*0+1)[0]
                tmp2 = self.fwd(psi+gammapsi * dpsi, scan, probe[:, 0]*0+1)[0]
                shifts = register_translation_batch(tmp1,tmp2,upsample_factor=100, space='fourier')
                #print(np.linalg.norm(shifts))
                scan[0,:]+=shifts
            # update psi
            psi = psi + gammapsi * dpsi


            
            if (recover_prb):
                if(i==0):
                    gradprb = probe*0
                    gradprb0 = probe*0
                    dprb = probe*0
                # if(ortho_prb):
                #     tt=orthogonalize_eig(probe[0].get())
                #   #  print(ids)
                #     probe[0]=cp.array(tt)
                    # dprb=dprb[:,ids]
                    # gradprb0=gradprb0[:,ids]    
                for m in range(0,probe.shape[1]):
                    # 2) probe retrieval subproblem with fixed object
                    # sum of forward operators associated with each probe                    
                    fprb = self.fwd(psi, scan, probe[:, m])
	                # sum of abs value of forward operators
                    absfprb = data*0
                    for k in range(probe.shape[1]):
                        tmp = self.fwd(psi, scan, probe[:, k])                        
                        absfprb += np.abs(tmp)**2                   
                    # take gradient
                    if model == 'gaussian':
                        gradprb[:,m] = self.adj_probe(
                            fprb - cp.sqrt(data) * fprb/(cp.sqrt(absfprb)+1e-32),
                            scan,
                            psi,
                        ) / cp.max(cp.abs(psi))**2 / self.nscan * probe.shape[1]#?
                    elif model == 'poisson':
                        gradprb[:,m] = self.adj_probe(
                            fprb - data * fprb / (absfprb + 1e-32),
                            scan,
                            psi,
                        ) / cp.max(cp.abs(psi))**2 / self.nscan
                    # Dai-Yuan direction
                    #dprb[:,m] = -gradprb[:,m]
                    if (i == 0):
                        dprb[:,m] = -gradprb[:,m]
                    else:
                        dprb[:,m] = -gradprb[:,m] + (
                            cp.linalg.norm(gradprb[:,m])**2 /
                            (cp.sum(cp.conj(dprb[:,m]) * (gradprb[:,m] - gradprb0[:,m]))) * dprb[:,m])
                    gradprb0[:,m] = gradprb[:,m]
                    # temp variables to avoid computing the fwd operator during the line serch
                    p1 = data*0
                    p2 = data*0
                    p3 = data*0
                    for k in range(probe.shape[1]):
                        tmp1 = self.fwd(psi, scan, probe[:, k])
                        p1 += cp.abs(tmp1)**2
                    tmp1 = self.fwd(psi, scan, probe[:, m])                        
                    tmp2 = self.fwd(psi, scan, dprb[:, m])                                                
                    p2 = cp.abs(tmp2)**2
                    p3 = 2*(tmp1.real*tmp2.real+tmp1.imag*tmp2.imag)
                    # line search		
                    gammaprb = 0.5*self.line_search_sqr(minf,p1,p2,p3,step_length=1)
                    # update probe                       
                    probe[:,m] = probe[:,m] + gammaprb * dprb[:,m]                
                              
                    # if(ortho_prb):
                    #     for k in range(m):
                    #         probe[:,m] = probe[:,m] \
                    #         - cp.sum(probe[:,m]*cp.conj(probe[:,k]))/cp.sum(probe[:,k]*cp.conj(probe[:,k]))*probe[:,k] 
                # probe=probe_new.copy()    
               
                        
            # check convergence
            if (np.mod(i, 32) == 0):
                sfpsi = cp.zeros(
                    [self.ptheta, self.nscan, self.ndet, self.ndet], dtype='complex64')
                for k in range(probe.shape[1]):
                    tmp = self.fwd(psi, scan, probe[:, k])
                    sfpsi += np.abs(tmp)**2
                print("%4d, %.3e, %.3e, %.7e" %
                      (i, gammapsi, gammaprb, minf(absfpsi)))
                #dxchange.write_tiff(cp.angle(psi[0]).get(),'tmp/'+str(i))

        return {
            'psi': psi,
            'probe': probe,
        }
