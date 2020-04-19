import dxchange 
import h5py
import numpy as np
for k in range(199,360):
    a = h5py.File('/local/data/vnikitin/catalyst/extracted_scan'+str(k)+'.h5','r')
    b = np.angle(a['matlab_obj_recon'][:])
    dxchange.write_tiff(b,'matlab/'+str(k))
# print(b.shape)
# for k in range(7):
#     print(np.linalg.norm(np.abs(a['recprobe'][k])))