from __future__ import print_function
import dxchange
import os
import scipy.misc
import matplotlib.pyplot as plt
import numpy as np
files = os.listdir('/home/beams/VNIKITIN/structured_illumination/ptychocg/psiang')
for name in files:
	print(name)
	a = dxchange.read_tiff('/home/beams/VNIKITIN/structured_illumination/ptychocg/psiang/'+name).copy()#[32:-32,32:-32].copy()	
	print(a.shape)
	scipy.misc.toimage(a, cmin=0, cmax=np.pi).save('png/'+str(os.path.splitext(name)[0])+'.png')

files = os.listdir('/home/beams/VNIKITIN/structured_illumination/ptychocg/psiamp')
for name in files:
	print(name)
	a = dxchange.read_tiff('/home/beams/VNIKITIN/structured_illumination/ptychocg/psiamp/'+name).copy()#[32:-32,32:-32].copy()	
	scipy.misc.toimage(a, cmin=0, cmax=1).save('png/'+str(os.path.splitext(name)[0])+'.png')
	# a[a<0.075]=0.075
	# a[a>0.97]=0.97
	# plt.imshow(a)#,cmap='hot')
	# plt.axis('off')
	# plt.savefig('png/'+str(os.path.splitext(name)[0])+'.png')

