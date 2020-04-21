import sys
import argparse
import logging
import re
import numpy as np
import h5py
import cupy as cp
import libtike.cufft as pt
import dxchange

class PtychoDAO(object):
  def __init__(self, 
      pid, data, positions, probes, 
      rotation_angle=None):
    self.pid = pid
    self.data = data
    self.positions = positions
    self.probes = probes
    self.rotation_angle = rotation_angle
  @classmethod
  def h5_reader(cls, 
                input_file, 
                pid=None,
                use_original_positions=False, 
                swap_position_axes=True,
                reset_position_coordinates=True,
                use_original_probes=False,
                swap_probe_axes=False,
                data_fftshift=True,
                view_dims=(2048,2048),
                map_position_detector_pixel=1.):
    logging.disable()
    logging.info("Processing input file:{}".format(input_file))
    if pid is None:
      # extract 339 from this '/data02/ptycho/extracted/lamni_33idd/extracted_scan339.h5'
      pid = np.int32(re.findall(r'\d+', input_file)[-2])
      logging.info("Id is not given, extracted from file: pid={}".format(pid))
    fid = h5py.File(input_file, 'r')

    # Arrange diffraction pattern data
    data = np.array(fid['data'], dtype=np.float32, order='C')
    logging.info("Diffraction pattern data was read: shape={}; type={}.".format(data.shape, data.dtype))
    # TODO: Below does not belong here and needs to be moved to compute kernels
    if data_fftshift: 
      logging.info("Diffraction pattern data was fftshifted.")
      data = np.fft.fftshift(data[:], axes=(1, 2))

    # Arrange position data
    detector_pixel_size = fid.attrs.get('detector_pixel_size')
    detector_distance = fid.attrs.get('detector_distance')
    incident_wavelength = fid.attrs.get('incident_wavelength')
    rotation_angle = fid.attrs.get('rotation_angle')

    logging.info("Metadata information was read: "
                  "rotation_angle={}; detector_pixel_size={}; "
                  "detector_distance={}; incident_wavelength={};".format(
                      rotation_angle, detector_pixel_size, detector_distance, incident_wavelength))
    # Use only the diffraction pattern data with the existing position information

    # Arrange probe data
    if use_original_probes: 
      probes = np.array(fid['/initprobe'], dtype=np.complex64, order='C')
      logging.info("Initial probes were read: shape={}; type={}".format(probes.shape, probes.dtype))
    else : 
      probes = np.array(fid['/recprobe'], dtype=np.complex64, order='C')
      logging.info("Reconstructed probes were read: shape={}; type={}".format(probes.shape, probes.dtype))
    if swap_probe_axes: 
      #probes = probes.swapaxes(1, 2) # WRONG!!
      probes = np.array(probes.swapaxes(1, 2),order='C')
      logging.info("Probe axes were swapped: new shape={}".format(probes.shape))

    if use_original_positions: 
      positions = np.array(fid['/positions_0'], dtype=np.float32, order='C')
      logging.info("Original positions were read: shape={}; type={}".format(positions.shape, positions.dtype))
    else : 
      positions = np.array(fid['/positions_1'], dtype=np.float32, order='C')
      logging.info("Real positions were read: shape={}; type={}".format(positions.shape, positions.dtype))
    pos2det_const = np.float64(((detector_pixel_size * probes.shape[-1]) / #XXX probes.shape[-1] is shaky
                      (detector_distance * 1e-10 * incident_wavelength)) * map_position_detector_pixel)
    positions = np.float32(positions * pos2det_const)
    logging.info("Positions were mapped to detector using: {}".format(pos2det_const))

    if swap_position_axes: 
      positions[:, (0,1)] = positions[:, (1,0)]
      logging.info("Positions' columns were swapped.")
    if reset_position_coordinates:
      positions[:, 0] = positions[:, 0] - min(positions[:, 0])
      positions[:, 1] = positions[:, 1] - min(positions[:, 1])
      ids = np.where( (positions[:, 1] >= 0) *
                      (positions[:, 1] < view_dims[1]) *
                      (positions[:, 0] >= 0) *
                      (positions[:, 0] < view_dims[0]) )[0]
      positions = np.array(positions[ids, :], dtype=np.float32, order='C')  # important!
      logging.info("Positions coordinates were setup to start from (0, 0).")
    else: 
      raise ValueException("Currently reset_position_coordinates has to be set to True.")

    if ids is not None: 
      data = data[ids] 
      logging.info("Diffraction pattern data was reorganized according to positions: new shape={}".format(data.shape))

    return cls(pid, data, positions, probes, rotation_angle)

def parse_arguments():
  parser = argparse.ArgumentParser(
          description='Ptychographic reconstruction with CG solver')
  parser.add_argument('--input_file', required=True,
                      help='Location of the ptychograpy data files. \
                            This file should have:\n \
                            diffraction patterns in /data;\n \
                            detector distance in detector_distance;\n \
                            detector pixel size in detector_pixel_size;\n \
                            incident wavelength in incident_wavelength;\n \
                            Original positions in positions_0;\n \
                            Real positions in positions_1;\n \
                            E.g. usage --path=\'/data02/ptycho/extracted/lamni_33idd/extracted_scan73.h5\'')
  #name = '/data01/ptycho/cSAXS_e18044_LamNI_201907/extracted/02661_out.h5' % (id)

  parser.add_argument('--niters', required=True, type=int,
                      help='Number of ptychography iterations.\
                            E.g. usage --nmodes=100')
  parser.add_argument('--nmodes', required=False, default=1, type=int,
                      help='Number of probe modes for reconstruction. Default value is 1.\
                            E.g. usage --nmodes=10')
  parser.add_argument('--nparallel', required=False, default=1, type=int,
                      help='Number of angles to process concurrently. Default value is 1.\
                            E.g. usage --nparallel=1')
  parser.add_argument('--gpu', required=False, default=0, type=int,
                      help='Sets the gpu to be used for reconstruction task. Default value is 0.\
                            E.g. usage --gpu=0')
  parser.add_argument('--model', required=False, default='gaussian',
                      help='Model to be used for reconstruction.Default value is gaussian\
                            Options are gaussian or poisson.\
                            E.g. usage --model=\'poisson\'')
  parser.add_argument('--probe_recovery', required=False, default=True, type=bool,
                      help='Recovers probe if set. Default value is True.\
                            E.g. usage --probe_recovery=True')
  parser.add_argument('--probe_ortho', required=False, default=1, type=int,
                      help='Orthogonalize probes after each iteration. Default value is 1.\
                            E.g. usage --probe_ortho=1')
  parser.add_argument('--output_prefix', required=False, default="rec_0",
                      help='Output file prefix. Default is rec_0. \
                      E.g. usage --output_prefix=\'rec_73\'.')
  parser.add_argument('--output_path', required=False, default="./rec",
                      help='Output folder path. Default is rec.\
                      E.g. usage --output_path=\'./rec\'.')

  parser.add_argument('--map_position_detector_pixel', required=False, default=1., type=np.float32,
                      help='Maps diffraction pattern position information to corresponding detector pixels.\
                            Reads incident_wavelength, detector_pixel_size and detector_distance \
                            from target input file, and computes the positions on detector pixel. \
                            Default value is used as a constant multiplier and set to 1.\
                            --map_position_detector_pixel_const overwrites this parameter.\
                            E.g. usage --map_position_detector_pixel=0.5')

  parser.add_argument('--view_dims', required=False, default='(2048,2048)',
                      help='Set the view dimensions.\
                      Note that the final view dimension is extended with the probe size.\
                      E.g. usage --view_dims=\'(2048,2048)\'.')

  parser.add_argument('--log', required=False, default="INFO",
                      help='Log level. E.g. usage --log=\'DEBUG\'. Default is INFO.')

  pargs = parser.parse_args()

  vdims = [np.int32(v) for v in re.findall(r'\d+', pargs.view_dims)]
  if len(vdims) != 2: raise ValueException("view_dims mush have two integers: {}".format(vdims))
  pargs.view_dims = tuple(vdims)

  return pargs

if __name__ == "__main__":
    args = parse_arguments() 
    logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
      datefmt='%Y-%m-%d:%H:%M:%S')
    logging.getLogger().setLevel(getattr(logging, args.log))
    cp.cuda.Device(args.gpu).use()  # gpu id to use
    # Read data
    view_dims = args.view_dims
    ptychoDAO = PtychoDAO.h5_reader(args.input_file, view_dims=view_dims)
    prb = ptychoDAO.probes
#    ids = np.array([0,1,2,3,5,6,4])
    #prb=prb[ids]
    
    
    prb.shape = (1, ) + prb.shape
    scan = ptychoDAO.positions
    print(scan.shape)
    # scan0=np.load('scan.npy')
    # print(scan.shape)

    # import matplotlib.pyplot as plt
    # plt.plot(scan[:,0],scan[:,1],'.')
    # plt.plot(scan0[:,0],scan0[:,1],'r.')
    # plt.show()
    # exit()



    scan.shape = (1, ) + scan.shape
    data = ptychoDAO.data
    data.shape = (1, ) + data.shape
    # Setup object
    psi = np.zeros((1, view_dims[0]+data.shape[-1], view_dims[1]+data.shape[-1]), 
                      dtype='complex64', order='C') + 1*np.exp(-1j*0.25)
    nmodes = args.nmodes
    prb = prb[:, :nmodes]    

    data/=np.amax(np.abs(prb))**2
    prb/=np.amax(np.abs(prb))


    [ntheta, nz, n] = psi.shape
    [ntheta, nscan, ndet, ndet] = data.shape
    nprb = prb.shape[2]
    ptheta = args.nparallel  # number of angles to process simultaneosuly
    model = args.model # minimization funcitonal (poisson,gaussian)
    piter = args.niters  # ptychography iterations
    logging.info("Number of angles={}; Number of scans={}; "
                  "Diff. pattern size={}x{}; Number of probes={}; "
                  "Parallelism={}".format(
                    ntheta, nscan, ndet, ndet, nprb, ptheta))
    logging.info("Probe shape={}; data shape={}; "
                  "scan shape={}; psi shape={}".format(
                    prb.shape, data.shape, scan.shape, psi.shape))
    recover_prb = True#args.probe_recovery  # recover probe or not
    ortho_prb = bool(args.probe_ortho)  # orthogonilize probe on each iteration or not

    with pt.CGPtychoSolver(nscan, nprb, ndet, ptheta, nz, n) as slv:
        result = slv.run_batch(
            data, psi, scan, prb, piter=piter, model=model, recover_prb=recover_prb, ortho_prb=ortho_prb)
        psi, prb = result['psi'], result['probe']
    # save result
    name = "{}_{}_{}_{}".format(args.output_prefix, str(model), str(nmodes), str(piter))
    hfile = args.output_path +"/{}.h5".format(name)
    spid = str(ptychoDAO.pid)
    with h5py.File(hfile, "w") as f:
      grp=f.create_group(spid)
      f.create_dataset(spid+"/psi", data=psi)
      f.create_dataset(spid+"/probe", data=prb)
      f[spid].attrs['rotation_angle'] = np.float32(ptychoDAO.rotation_angle)
      logging.info("File ({}) is updated with pid={}".format(hfile, spid))
    dxchange.write_tiff_stack(np.angle(psi),
                        args.output_path + "/" + spid + '/psi_angle/' + name + '.tiff',overwrite=True)
    dxchange.write_tiff_stack(np.abs(psi),  
                        args.output_path + "/" + spid + '/psi_amp/' + name + '.tiff', overwrite=True)
    dxchange.write_tiff_stack(np.angle(prb[0]),
                        args.output_path + "/" + spid + '/probe_angle/' + name + '.tiff', overwrite=True)
    dxchange.write_tiff_stack(np.abs(prb[0]), 
                        args.output_path + "/" + spid + '/probe_amp/' + name + '.tiff', overwrite=True)
