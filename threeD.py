import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
 
delta = 0.2
x = np.arange(-3, 3, delta)
y = np.arange(-3, 3, delta)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2
x=X.flatten()
y=Y.flatten()
z=Z.flatten()
 
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_trisurf(x, y, z, cmap=cm.jet, linewidth=0.01)
plt.figure(2)
plt.subplot(111)
plt.contourf(X,Y,Z,cmap=cm.jet,alpha=0.87)

#sigmoid 函数
import matplotlib.pyplot as plt
import numpy as np
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))
z=np.arange(-7,7,0.1)
plt.figure(3)
plt.subplot(111)
plt.plot(z, sigmoid(z))
#加垂直线标注
plt.axvline(0.0 , color = 'k')
#加水平线
plt.axhspan(0.0 , 1.0 , facecolor = '1.0' ,alpha = 0.6 , ls = 'dotted')
plt.axhline(y=0.5 , ls = 'dotted' ,color = 'k')
#y轴显示数据
plt.yticks([0.0 , 0.5 , 1.0])
plt.ylim(-0.1, 1.1)

plt.show()