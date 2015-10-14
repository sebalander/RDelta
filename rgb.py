
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt


im=plt.imread('papel3.jpg')
s = im.shape
l = s[0]*s[1]
[r, g, b] = im.reshape([l,s[2]]).transpose()





fig = plt.figure()
ax = fig.gca(projection='3d')

Dl=15 # grafico cada Dl
ax.scatter(r[0:l:Dl], g[0:l:Dl], b[0:l:Dl], s=0.1)

ax.set_xlim3d(0, 255)
ax.set_ylim3d(0, 255)
ax.set_zlim3d(0, 255)

ax.set_xlabel('Rojo')
ax.set_ylabel('Verde')
ax.set_zlabel('Azul')

plt.show()


