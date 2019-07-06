from __future__ import print_function
import dxchange
import os
import scipy.misc

files = os.listdir('betap')
for name in files:
	print(name)
	a = dxchange.read_tiff('betap/'+name)
	print(a.shape)
	scipy.misc.toimage(a, cmin=0, cmax=0.0000311).save('png/beta/'+str(os.path.splitext(name)[0])+'.png')

files = os.listdir('deltap')
for name in files:
	a = dxchange.read_tiff('deltap/'+name)+0.0000183
	scipy.misc.toimage(a, cmin=-0.0000508+0.0000183, cmax=0.000127+0.0000183).save('png/delta/'+str(os.path.splitext(name)[0])+'.png')


