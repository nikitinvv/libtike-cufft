import dxchange
import numpy as np
import sys
import os


if __name__ == "__main__":
    name = sys.argv[1]
    a = dxchange.read_tiff(name)
    bg = np.mean(a[4:16],axis=0)#.5*(np.mean(a[12:24],axis=0)+np.mean(a[512-24:512-12],axis=0))
    b = a-bg
    c = os.path.splitext(name)[0]
    dxchange.write_tiff(b,c+'c')