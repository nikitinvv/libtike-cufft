import dxchange
import numpy as np

noise = True
prbid = 1
prbshift = 64
model = 'gaussian'
prbfile = 'probes-0.1'
ang0 = dxchange.read_tiff('anginit.tiff')

for prbid in range(1,4):
   # Save result
    name = str(prbfile)+'prbid' + \
        str(prbid)+'prbshift'+str(prbshift)+str(model)+'.tiff'
    ang = dxchange.read_tiff('psiang/psiang'+name).copy()

    ang+=np.pi
    ang += 2*np.pi*np.int32((ang0-ang-np.pi)/(2*np.pi))
    dxchange.write_tiff(ang,  'psiang/psiang_un_'+name, overwrite=True)
