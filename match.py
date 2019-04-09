import provider
import os
import numpy as np
from collections import Counter
from random import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_FILES = provider.getDataFiles( \
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))
train_file_idxs = np.arange(0, len(TRAIN_FILES))
NUM_POINT = 2048
# put things together
data = []
label = []
for fn in range(len(TRAIN_FILES)):
    current_data, current_label = provider.loadDataFile(TRAIN_FILES[train_file_idxs[fn]])
    current_data = current_data[:, 0:NUM_POINT, :]
    data.append(current_data)
    label.append(current_label)
templen = 0
for l1 in range(len(data)):
    templen += data[l1].shape[0]
data_complete = np.ndarray(shape=(templen, NUM_POINT, 3), dtype=float, order='F')
label_complete = np.ndarray(shape=(templen, 1), dtype=float, order='F')
counter = 0
for i in range(len(data)):
    for j in range(len(data[i])):
        data_complete[counter] = data[i][j]
        label_complete[counter] = label[i][j]
        counter += 1
label_complete = np.squeeze(label_complete)
# find the biggest type
type_count = Counter(label_complete)
most_common = type_count.most_common(40)
print(most_common)
length = most_common[0][1]
pick_ones = np.ndarray(shape=(length, NUM_POINT, 3), dtype=float, order='F')
counter = 0
for i in range(len(label_complete)):
    if label_complete[i] == 8.0:
        pick_ones[counter] = data_complete[i]
        counter += 1
for i in range(len(pick_ones)):
    pick_points = data_complete[i]
    x = pick_points[:, 0]
    y = pick_points[:, 1]
    z = pick_points[:, 2]
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x, y, z)
    plt.title('raw points')
    plt.show()
pick = randint(0, counter - 1)
pick_points = data_complete[pick]
x = pick_points[:, 0]
y = pick_points[:, 1]
z = pick_points[:, 2]
fig = plt.figure()

ax = Axes3D(fig)
ax.scatter(x, y, z)
plt.title('raw points')
plt.show()

tri = Delaunay(np.array([x, y]).T)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.plot_trisurf(x, y, z, triangles=tri.simplices, cmap=plt.cm.Spectral)
plt.title('Delaunay mesh')
plt.show()
