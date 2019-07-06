from __future__ import print_function
import dxchange
import os
import numpy as np
import sys

if __name__ == "__main__":
    name = sys.argv[1]
    a = dxchange.read_tiff(name)
    size = 10
    left = np.int(512/10.0*3.5)+20
    bottom = np.int(512/10.0*4.55)-3
    right = np.int(512/10.0*5.0)-14
    top = np.int(512/10.0*3.95)+11
    print(a[45:45+70,top:512-bottom,left:512-right].shape)
    dxchange.write_tiff(a[45:45+70,top:512-bottom,left:512-right],os.path.splitext(name)[0]+'c',overwrite=True)
