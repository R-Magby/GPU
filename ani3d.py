import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib import pyplot as plt
from matplotlib import cm
from IPython import display

import codecs
from random import choice
from typing import List, Any

data=np.loadtxt("onda_2d_cuda.dat").reshape((10000,64,64))
 
X=list(range(64))
Y=list(range(64))

xx,yy=np.meshgrid(X,Y)

fig, axl = plt.subplots(subplot_kw=dict(projection='3d'))
#line, = axl.plot(X,data[0,:],"-k" )

def actualizarL(i):
    axl.clear()
    axl.set_zlim(-3, 3)
    surf = axl.plot_surface(xx, yy, data[i*10,:,:], cmap=cm.coolwarm,linewidth=0, antialiased=False)


chu=FuncAnimation(fig,actualizarL,interval=20,frames=5000)
#chu.save('onda2d_cuda.gif')
plt.show()
# good practice to close the plt object.
