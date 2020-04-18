import dxchange 
import h5py
import numpy as np
a = h5py.File('extracted_scan201.h5','r')
b = np.abs(a['recprobe'][:])
dxchange.write_tiff(b,'recprobe')
print(b.shape)
for k in range(7):
    print(np.linalg.norm(np.abs(a['recprobe'][k])))