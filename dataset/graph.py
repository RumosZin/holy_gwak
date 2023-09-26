import numpy as np
import matplotlib.pyplot as plt
import sys
from mpl_toolkits import mplot3d

# np.set_printoptions(threshold=sys.maxsize)

fig = plt.figure()
ax = plt.axes(projection='3d')

vx_CIRCLE_1 = np.load('vx_CIRCLE_1.npy')
vy_CIRCLE_1 = np.load('vy_CIRCLE_2.npy')
vz_CIRCLE_1 = np.load('vz_CIRCLE_3.npy')

vx_CIRCLE_2 = np.load('vx_CIRCLE_2.npy')
vy_CIRCLE_2 = np.load('vy_CIRCLE_2.npy')
vz_CIRCLE_2 = np.load('vz_CIRCLE_2.npy')

vx_CIRCLE_3 = np.load('vx_CIRCLE_3.npy')
vy_CIRCLE_3 = np.load('vy_CIRCLE_3.npy')
vz_CIRCLE_3 = np.load('vz_CIRCLE_3.npy')

vx_HORIZONTAL_1 = np.load('vx_HORIZONTAL_1.npy')
vy_HORIZONTAL_1 = np.load('vy_HORIZONTAL_1.npy')
vz_HORIZONTAL_1 = np.load('vz_HORIZONTAL_1.npy')

vx_HORIZONTAL_2 = np.load('vx_HORIZONTAL_2.npy')
vy_HORIZONTAL_2 = np.load('vy_HORIZONTAL_2.npy')
vz_HORIZONTAL_2 = np.load('vz_HORIZONTAL_2.npy')

vx_HORIZONTAL_3 = np.load('vx_HORIZONTAL_3.npy')
vy_HORIZONTAL_3 = np.load('vy_HORIZONTAL_3.npy')
vz_HORIZONTAL_3 = np.load('vz_HORIZONTAL_3.npy')

vx_HORIZONTAL_4 = np.load('vx_HORIZONTAL_4.npy')
vy_HORIZONTAL_4 = np.load('vy_HORIZONTAL_4.npy')
vz_HORIZONTAL_4 = np.load('vz_HORIZONTAL_4.npy')

vx_VERTICAL_1 = np.load('vx_VERTICAL_1.npy')
vy_VERTICAL_1 = np.load('vy_VERTICAL_1.npy')
vz_VERTICAL_1 = np.load('vz_VERTICAL_1.npy')

vx_VERTICAL_2 = np.load('vx_VERTICAL_2.npy')
vy_VERTICAL_2 = np.load('vy_VERTICAL_2.npy')
vz_VERTICAL_2 = np.load('vz_VERTICAL_2.npy')

vy_CIRCLE_1 = vy_CIRCLE_1[:74]
vz_CIRCLE_1 = vz_CIRCLE_1[:74]
print(vx_CIRCLE_1.shape)
print(vy_CIRCLE_1.shape)
print(vz_CIRCLE_1.shape)

# plot x, y, z
plt.figure(figsize=(12, 4))
ax.scatter3D(vx_CIRCLE_1, vy_CIRCLE_1, vz_CIRCLE_1, marker='o', color='red')
ax.scatter3D(vx_CIRCLE_2, vy_CIRCLE_2, vz_CIRCLE_2, marker='o', color='pink')
ax.scatter3D(vx_CIRCLE_3, vy_CIRCLE_3, vz_CIRCLE_3, marker='o', color='orange')
ax.scatter3D(vx_HORIZONTAL_1, vy_HORIZONTAL_1, vz_HORIZONTAL_1, marker='o', color='blue')
ax.scatter3D(vx_HORIZONTAL_2, vy_HORIZONTAL_2, vz_HORIZONTAL_2, marker='o', color='green')
ax.scatter3D(vx_HORIZONTAL_3, vy_HORIZONTAL_3, vz_HORIZONTAL_3, marker='o', color='lime')
ax.scatter3D(vx_HORIZONTAL_4, vy_HORIZONTAL_4, vz_HORIZONTAL_4, marker='o', color='yellow')
ax.scatter3D(vx_VERTICAL_1, vy_VERTICAL_1, vz_VERTICAL_1, marker='o', color='dodgerblue')
ax.scatter3D(vx_VERTICAL_2, vy_VERTICAL_2, vz_VERTICAL_2, marker='o', color='magenta')

plt.show()