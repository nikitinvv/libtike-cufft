import dxchange 
import h5py
import numpy as np
from scipy import ndimage
ang = np.zeros(168,dtype='float32')
rec = np.zeros([168,512,512],dtype='float32')
j=0
ba = dxchange.read_tiff('/home/beams/VNIKITIN/lamino_doga/libtike-cufft/tests/catalyst/resall.tif',)
for k in range(192,361):
    if(k==198):
        continue
    a = h5py.File('/local/data/vnikitin/catalyst/extracted_scan'+str(k)+'.h5','r')
    ang[j]= a.attrs['rotation_angle']
    tilt = a.attrs['tilt_angle']
    b = np.angle(a['matlab_obj_recon'][:])
    b = ba[j]
    b = ndimage.interpolation.rotate(b,-tilt,order=5)
    c=ndimage.measurements.center_of_mass(b*(b>0))
    cx = np.int(c[1])
    cy = np.int(c[0])
    b=b[cy-350:cy+350,cx-350:cx+350]    
    b=b+0.305
    c=ndimage.measurements.center_of_mass(b*(b>0))
    cx = np.int(c[1])
    cy = np.int(c[0])    
    b=b[cy-256:cy+256,cx-256:cx+256]
    b-=np.mean(b[32:64,32:64])
    rec[j,0:b.shape[0],0:b.shape[1]]=b
    j+=1
    # dxchange.write_tiff(b,'matlab/'+str(k),overwrite=True)
    # dxchange.write_tiff(,'matlaba/'+str(k),overwrite=True)
ids = np.argsort(ang)    
rec = rec[ids]
dxchange.write_tiff(rec,'rec_new',overwrite=True)
np.save('angle',ang[ids])
# print(b.shape)
# for k in range(7):
#     print(np.linalg.norm(np.abs(a['recprobe'][k])))