#产生异或
#np.logical_xor(x1,x2)
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
#from scikit_learn_test import plot_decision_regions

def plot_decision_regions(x,y,classifier,test_idx=None,resolution=0.02):
    #setup marker generator and color map
    markers = ('s','x','o','^','v')
    colors = ('red','blue','green','gray','cyan')
    cmap=ListedColormap(colors[:len(np.unique(y))])
   
    #plot the decision surface
    x1_min, x1_max = x[:,0].min() - 1, x[:,0].max() + 1
    x2_min, x2_max = x[:,1].min() - 1, x[:,0].max() + 1
    #meshgrid坐标 xx1 : x , xx2 : y*
    xx1, xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution),
                            np.arange(x2_min,x2_max,resolution))
    z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    z = z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, z, alpha = 0.1, cmap = cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
   
    #plot all samples
    x_test,y_test = x[test_idx,:], y[test_idx]
    for idx,cl in enumerate(np.unique(y)):
        plt.scatter(x = x[y == cl, 0],y = x[y == cl, 1],
                    alpha=0.6,c = cmap(idx),
                    marker = markers[idx], label = cl)
   
    #highlight test samples
    #scatter(x,y,color,alpha()透明度,linewidth(线宽),marker,s(散点大小))
 
    if test_idx:
        x_test,y_test=x[test_idx,:],y[test_idx]
        plt.scatter(x_test[:,0],x_test[:,1],
                    c='k',alpha=1.0,linewidths=1,
                    marker='o',s=5,label='test_set')


np.random.seed(0)
x_xor = np.random.randn(200,2)
y_xor1 = np.logical_xor(x_xor[:,0] > 0 , x_xor[:,1] > 0)
y_xor = np.where(y_xor1, 1, -1)




svm = SVC(kernel= 'rbf', random_state= 0, gamma= 0.10, C= 10.0)
svm.fit(x_xor, y_xor)
plt.figure(2)
plt.subplot(111)
plot_decision_regions(x_xor, y_xor, classifier= svm)
plt.ylim(-3.0)
plt.legend(loc= 'upper left')

plt.figure(1)
plt.subplot(111)
plt.scatter(x_xor[y_xor == 1, 0], x_xor[y_xor == 1, 1], 
            c = 'b', marker= 'x', label ='1')
plt.scatter(x_xor[y_xor == -1, 0], x_xor[y_xor == -1, 1],
            c= 'r', marker= 'o', label= '-1')
plt.ylim(-3.0)
plt.legend()
plt.show()
