from __future__ import print_function
import dxchange
import os
import scipy.misc
import matplotlib.pyplot as plt
# files = os.listdir('/home/beams/VNIKITIN/structured_illumination/ptychotomo/psiangle')
# for name in files:
# 	print(name)
# 	a = dxchange.read_tiff('/home/beams/VNIKITIN/structured_illumination/ptychotomo/psiangle/'+name)[64:-64,64:-64]	
# 	print(a.shape)
# 	scipy.misc.toimage(a+1, cmin=-0.9, cmax=1.075).save('png/'+str(os.path.splitext(name)[0])+'.png')

files = os.listdir('/home/beams/VNIKITIN/structured_illumination/ptychotomo/psiangle')
for name in files:
	print(name)
	a = dxchange.read_tiff('/home/beams/VNIKITIN/structured_illumination/ptychotomo/psiangle/'+name).copy()#[32:-32,32:-32].copy()	
	print(a.shape)
	a+=1
	scipy.misc.toimage(a, cmin=-0.9, cmax=1.075).save('png/'+str(os.path.splitext(name)[0])+'.png')
	
	# a[a<-0.9]=-0.9
	# a[a>1.075]=1.075
	# plt.imshow(a)
	# plt.axis('off')
	# plt.savefig('png/'+str(os.path.splitext(name)[0])+'.png')



files = os.listdir('/home/beams/VNIKITIN/structured_illumination/ptychotomo/psiamp')
for name in files:
	print(name)
	a = dxchange.read_tiff('/home/beams/VNIKITIN/structured_illumination/ptychotomo/psiamp/'+name).copy()#[32:-32,32:-32].copy()	
	print(a.shape)
	scipy.misc.toimage(a, cmin=0.075, cmax=1.1).save('png/'+str(os.path.splitext(name)[0])+'.png')
	# a[a<0.075]=0.075
	# a[a>0.97]=0.97
	# plt.imshow(a)#,cmap='hot')
	# plt.axis('off')
	# plt.savefig('png/'+str(os.path.splitext(name)[0])+'.png')

