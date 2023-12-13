import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib import pyplot as plt
from matplotlib import cm

import codecs
from random import choice
from typing import List, Any
data=np.array([])
n=10
for i in range(n):
    data0=np.fromfile("n_body"+str(i)+".dat",dtype=np.float32)
    data=np.concatenate([data,data0],axis=None)



largo=int(len(data)/(3*500*n))
data=data.reshape((500*n,largo,3))




fig, axl = plt.subplots(subplot_kw=dict(projection='3d'),facecolor='black')

axl.set_facecolor("black")
#line, = axl.scatter(X,data[0,:],"-k" )
print(len(data),len(data[0,:,:]))
print(data[98:105,0,:])
def actualizarL(i):
    axl.clear()
    axl.set_xlim(0, 512)
    axl.set_ylim(0, 512)
    axl.set_zlim(0, 512)
    #axl.set_axis_off()
    axl.scatter(data[i*10,:,0], data[i*10,:,1], data[i*10,:,2], c="red",label=i*500,s=5.5,linewidth=0, antialiased=False)
    axl.legend()


chu=FuncAnimation(fig,actualizarL,interval=1,frames=500*n)
#chu.save('ondas_stream2.gif')
plt.show()
# good practice to close the plt object.