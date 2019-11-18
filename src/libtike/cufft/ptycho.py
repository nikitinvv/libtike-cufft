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

from libtike.cufft.ptychofft import ptychofft


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
    ptheta : int
        The number of angular partitions of the data.
    n, nz : int
        The pixel width and height of the reconstructed grid.
    ptheta : int
        The number of angular partitions to process together
        simultaneously.
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
        ptychofft.fwd(self, farplane.data.ptr, psi.data.ptr, scan.data.ptr, probe.data.ptr)
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
        ptychofft.adj(self, psi.data.ptr, farplane.data.ptr, scan.data.ptr, probe.data.ptr, flg)
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
        probe = cp.zeros([self.ptheta, self.nprb, self.nprb], dtype='complex64')
        flg = 1  # compute adjoint operator with respect to probe
        ptychofft.adj(self, psi.data.ptr, farplane.data.ptr, scan.data.ptr, probe.data.ptr, flg)
        return probe

    def adj_ptycho_batch_prb(self, farplane, scan, psi):
        """Batch of Ptychography transform (FQ)."""
        probe = np.zeros([scan.shape[0], self.nprb, self.nprb], dtype='complex64')
        return self._batch(self.adj_probe, probe, farplane, scan, psi)

    def run(self, data, psi, scan, probe, **kwargs):
        """Placehold for a child's solving function."""
        raise NotImplementedError("Cannot run a base class.")

    def run_batch(self, data, psi, scan, probe, **kwargs):
        """Run by dividing the work into batches."""
        assert probe.ndim == 3, "probe needs 3 dimensions, not %d" % probe.ndim

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


class CGPtychoSolver(PtychoCuFFT):
    """Solve the ptychography problem using congujate gradient."""

    @staticmethod
    def line_search(f, x, d, step_length=1, step_shrink=0.5):
        """Return a new step_length using a backtracking line search.

        https://en.wikipedia.org/wiki/Backtracking_line_search

        Parameters
        ----------
        f : function(x)
            The function being optimized.
        x : vector
            The current position.
        d : vector
            The search direction.

        """
        assert step_shrink > 0 and step_shrink < 1
        m = 0  # Some tuning parameter for termination
        fx = f(x)  # Save the result of f(x) instead of computing it many times
        # Decrease the step length while the step increases the cost function
        while f(x + step_length * d) > fx + step_shrink * m:
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
        assert probe.ndim == 3, "probe needs 3 dimensions, not %d" % probe.ndim

        # minimization functional
        def minf(fpsi):
            if model == 'gaussian':
                f = cp.linalg.norm(cp.abs(fpsi) - cp.sqrt(data))**2
            elif model == 'poisson':
                f = cp.sum(
                    cp.abs(fpsi)**2 - 2 * data * cp.log(cp.abs(fpsi) + 1e-32))
            return f

        print("# congujate gradient parameters\n"
              "iteration, step size object, step size probe, function min"
              )  # csv column headers
        gammaprb = 0
        for i in range(piter):
            # 1) object retrieval subproblem with fixed probe
            # forward operator
            fpsi = self.fwd(psi, scan, probe)
            # take gradient
            if model == 'gaussian':
                gradpsi = self.adj(
                    fpsi - cp.sqrt(data) * cp.exp(1j * cp.angle(fpsi)),
                    scan,
                    probe,
                ) / (cp.max(cp.abs(probe))**2)
            elif model == 'poisson':
                gradpsi = self.adj(
                    fpsi - data * fpsi / (cp.abs(fpsi)**2 + 1e-32),
                    scan,
                    probe,
                ) / (cp.max(cp.abs(probe))**2)
            # Dai-Yuan direction
            if i == 0:
                dpsi = -gradpsi
            else:
                dpsi = -gradpsi + (
                    cp.linalg.norm(gradpsi)**2 /
                    (cp.sum(cp.conj(dpsi) * (gradpsi - gradpsi0))) * dpsi)
            gradpsi0 = gradpsi
            # line search
            fdpsi = self.fwd(dpsi, scan, probe)
            gammapsi = self.line_search(minf, fpsi, fdpsi)
            # update psi
            psi = psi + gammapsi * dpsi

            if (recover_prb):
                # 2) probe retrieval subproblem with fixed object
                # forward operator
                fprb = self.fwd(psi, scan, probe)
                # take gradient
                if model == 'gaussian':
                    gradprb = self.adj_probe(
                        fprb - cp.sqrt(data) * cp.exp(1j * cp.angle(fprb)),
                        scan,
                        psi,
                    ) / cp.max(cp.abs(psi))**2 / self.nscan
                elif model == 'poisson':
                    gradprb = self.adj_probe(
                        fprb - data * fprb / (cp.abs(fprb)**2 + 1e-32),
                        scan,
                        psi,
                    ) / cp.max(cp.abs(psi))**2 / self.nscan
                # Dai-Yuan direction
                if (i == 0):
                    dprb = -gradprb
                else:
                    dprb = -gradprb + (
                        cp.linalg.norm(gradprb)**2 /
                        (cp.sum(cp.conj(dprb) * (gradprb - gradprb0))) * dprb)
                gradprb0 = gradprb
                # line search
                fdprb = self.fwd(psi, scan, dprb)
                gammaprb = self.line_search(minf, fprb, fdprb)
                # update probe
                probe = probe + gammaprb * dprb

            # check convergence
            if (np.mod(i, 8) == 0):
                fpsi = self.fwd(psi, scan, probe)
                print("%4d, %.3e, %.3e, %.7e" %
                      (i, gammapsi, gammaprb, minf(fpsi)))

        return {
            'psi': psi,
            'probe': probe,
        }
