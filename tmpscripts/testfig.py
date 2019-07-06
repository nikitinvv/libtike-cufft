import solver
import dxchange
import objects
import numpy as np
import cupy as cp
import signal
import sys


if __name__ == "__main__":

    igpu = np.int(sys.argv[1])
    cp.cuda.Device(igpu).use()  # gpu id to use
    
    # Model parameters
    voxelsize = 1e-6  # object voxel size
    energy = 5  # xray energy
    maxinta = [3]  # maximal probe intensity
    prbsize = 16 # probe size
    prbshift = 8  # probe shift (probe overlap = (1-prbshift)/prbsize)
    det = [64, 64] # detector size
    ntheta = 256*3//2  # number of angles (rotations)
    noisea = [False]  # apply discrete Poisson noise

    # Reconstrucion parameters
    modela = ['poisson','gaussian']  # minimization funcitonal (poisson,gaussian)
    alphaa = [1e-12] # tv regularization penalty coefficient    
    
    print(igpu,alphaa)
    piter = 4  # ptychography iterations
    titer = 4  # tomography iterations
    NITER = 500  # ADMM iterations

    # Load a 3D object
    beta = dxchange.read_tiff('data/BETA256.tiff')[32:80]
    delta = dxchange.read_tiff('data/DELTA256.tiff')[32:80]


    for imaxint in range(0,len(maxinta)):
        for inoise in range(0,len(noisea)):
            maxint = maxinta[imaxint]
            noise = noisea[inoise]
            
            # Create object, probe, angles, scan positions
            obj = cp.array(delta+1j*beta)
            prb = cp.array(objects.probe(prbsize, maxint))
            theta = cp.linspace(0, np.pi, ntheta).astype('float32')
            scan = cp.array(objects.scanner3(theta, obj.shape, prbshift,
                                            prbshift, prbsize, spiral=0, randscan=True, save=True)) 
            tomoshape = [len(theta), obj.shape[0], obj.shape[2]]

            # Class gpu solver 
            slv = solver.Solver(prb, scan, theta, det, voxelsize, energy, tomoshape)
            # Free gpu memory after SIGINT, SIGSTSTP
            def signal_handler(sig, frame):
                slv = []
                sys.exit(0)
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTSTP, signal_handler)

            
            # Compute data
            psi = slv.exptomo(slv.fwd_tomo(obj))
            data = np.zeros(slv.ptychoshape, dtype='float32')
            for k in range(0, 16):  # angle partitions in ptyocgraphy
                ids = np.arange(k*ntheta//16, (k+1)*ntheta//16)
                slv.cl_ptycho.setobj(scan[:, ids].data.ptr, prb.data.ptr)
                data[ids] = (cp.abs(slv.fwd_ptycho(psi[ids]))**2/slv.coefdata).get()
            print("max data = ", np.amax(data))      
            if (noise == True):# Apply Poisson noise
                data = np.random.poisson(data).astype('float32')
            
            for ialpha in range(0,len(alphaa)):            
                for imodel in range(0,len(modela)):
                    model = modela[imodel]
                    alpha = alphaa[ialpha]                    
                    # Initial guess
                    h = cp.zeros(tomoshape, dtype='complex64', order='C')+1
                    psi = cp.zeros(tomoshape, dtype='complex64', order='C')+1
                    e = cp.zeros([3, *obj.shape], dtype='complex64', order='C')
                    phi = cp.zeros([3, *obj.shape], dtype='complex64', order='C')
                    lamd = cp.zeros(tomoshape, dtype='complex64', order='C')
                    mu = cp.zeros([3, *obj.shape], dtype='complex64', order='C')
                    u = cp.zeros(obj.shape, dtype='complex64', order='C')
                    # ADMM
                    u, psi, lagr = slv.admm(data, h, e, psi, phi, lamd,
                                        mu, u, alpha, piter, titer, NITER, model)

                    # Save result
                    name = 'reg'+str(alpha)+'noise'+str(noise)+'maxint' + \
                        str(maxint)+'prbshift'+str(prbshift)+'ntheta'+str(ntheta)+str(model)+str(piter)+str(titer)+str(NITER)
                    print(name)
                    dxchange.write_tiff(u.imag.get(),  'beta3/beta'+name)
                    dxchange.write_tiff(u.real.get(),  'delta3/delta'+name)
                    dxchange.write_tiff(u[u.shape[0]//2].imag.get(),  'betaslicefig/beta'+name)
                    dxchange.write_tiff(u[u.shape[0]//2].real.get(),  'deltaslicefig/delta'+name)
                    dxchange.write_tiff(psi.imag.get(),  'psi3/psii'+name)
                    dxchange.write_tiff(psi.real.get(),  'psi3/psir'+name)    
                    np.save('lagr3/lagr'+name,lagr.get())

