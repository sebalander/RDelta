from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.vq import vq, kmeans#, whiten

# cargo imagen
im = plt.imread('papel3.jpg');
s = im.shape; # tamaño
l = s[0]*s[1]; # cant de pixeles

# saco los datos, solo RGB de todos los pixeles
data =  im.reshape([l,s[2]]);


# los lugares aproximados de los cumulos
#            R     G      B
#Marrón      41    23     18
#Bordó      102     5     26
#Rojo       176     0     28
#Amarillo   173   133      0
#Verde        0   110      0
#Azul        17    62    104
#Papel      177   167    104



# ============ entreno al algoritmo============
# posiciones iniciales
#            R     G     B
k= np.array([[41 , 23 , 18],#Marrón
            [102 , 5 , 26],#Bordó
            [176 , 0 , 28],#Rojo
            [173 , 133 , 0],#Amarillo
            [0 , 110 , 0],#Verde
            [17 , 62 , 104],#Azul
            [177 , 167 , 104],#Papel
            [160 , 15 , 90]]);#Papel 2, porque es un cluster muy extenso

# tipear "help(kmeans)" por cualquier cosa
[centroids, distortion] = kmeans(data,k);

# clasifico usando la funcion vector quantization ("help(vq)")
[clases, distortion] = vq(data,centroids);


# ordeno (traspongo) para graficar
[cr, cg, cb] = centroids.transpose();
[r, g, b] = data.transpose();
CenCols = centroids/256.0; # color de cada centroide
Cols = CenCols[clases]; # a cada dato el color de su centroide



# ============ grafico ============
# ============ cumulos 
fig = plt.figure();
ax = fig.gca(projection='3d');

Dl=15; # grafico cada Dl pixeles, para no hacer todos, que son muchos

# los datos con los colores de su centroide
ax.scatter(r[0:l:Dl], g[0:l:Dl], b[0:l:Dl], 
    marker='.', # marcadores tipo punto
    c=Cols[0:l:Dl], # color
    linewidth=0, # marcadores sin bordes
    alpha=0.1); # trasnparencia

# grafico centroides como estrellas
ax.scatter(cr, cg, cb, 
    s=150, # tamaño grande
    marker='*', # estrellaº
    c=CenCols); # colores

ax.set_xlim3d(0, 255);
ax.set_ylim3d(0, 255);
ax.set_zlim3d(0, 255);

ax.set_xlabel('Rojo');
ax.set_ylabel('Verde');
ax.set_zlabel('Azul');

plt.show();

# ============ imagen segmentada

im2 = np.reshape(clases,s[0:2]) # le doy forma de imagen
plt.imshow(im2)
plt.show()




