import numpy as np

noise = False # True, False
prbshift = 10 # 10, 4
ntheta = 400 # 100, 400

name = 'noise'+str(noise)+'prbshift'+str(prbshift)+'ntheta'+str(ntheta)+'.npy'

data = np.load('wires_data/data'+name) # shape: (number of angles, number of scans per projection, detectors size x, detector size y)
coordinates = np.load('wires_data/coordinates'+name) # shape: (2:(x,y)-scan position, number of angles, number of scans per projection)
theta = np.load('wires_data/theta'+name) # shape: (number of angles)
prb = np.load('wires_data/prb'+name) # shape: (probe size x, probe size y), complex array

for j in range(0,8):
    fig, ax = plt.subplots(1)
    plt.xlim(0, 64)
    plt.ylim(0, 64)
    plt.gca().set_aspect('equal', adjustable='box')
    for k in range(0, scan.shape[2]):
        c = patches.Circle(
            (scana[0,j,k]+prb.shape[0]//2, scana[0,j,k]+prb.shape[0]//2, prb.shape[0], fill=False, edgecolor=[*random_color(),1])
        ax.add_patch(c)
    plt.savefig('scan'+str(j)+'.png')
print(data.shape)
print(coordinates.shape)
print(theta.shape)
print(prb.shape)
