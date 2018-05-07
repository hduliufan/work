#数据集scikitlearn自带的iris数据
from sklearn import datasets
import numpy as np
iris=datasets.load_iris()
x = iris.data[:, [2,3]]
#一维数组
y = iris.target
#数据集分类，训练集train-data，测试集test-data
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=0)
#数据标准化standardized
#训练集和测试集有同样的均值和方差
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaler = scaler.transform(x_train)
x_test_scaler = scaler.transform(x_test)
from sklearn.linear_model import Perceptron
ppn = Perceptron(eta0=0.01,random_state=0,n_iter=40)
ppn.fit(x_train_scaler,y_train)

y_predict = ppn.predict(x_test_scaler)
print('misclassified samples:%d'%(y_predict!=y_test).sum())
print('classified rata:%.2f'%(1-(y_predict!=y_test).sum()/len(y_test)))
from sklearn.metrics import accuracy_score
print('classified rata:%.2f'%accuracy_score(y_test,y_predict))
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
def plot_decision_regions(x,y,classifier,test_idx=None,resolution=0.02):
    #setup marker generator and color map
    markers = ('s','x','o','^','v')
    colors = ('red','blue','green','gray','cyan')
    cmap=ListedColormap(colors[:len(np.unique(y))])
   
    #plot the decision surface
    x1_min, x1_max = x[:,0].min() - 1, x[:,0].max() + 1
    x2_min, x2_max = x[:,1].min() - 1, x[:,0].max() + 1
    #meshgrid坐标 xx1 : x , xx2 : y
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
    '''
    scatter(x,y,color,alpha()透明度,linewidth(线宽),marker,s(散点大小))
    '''
    if test_idx:
        x_test,y_test=x[test_idx,:],y[test_idx]
        plt.scatter(x_test[:,0],x_test[:,1],
                    c='Tan',alpha=1.0,linewidths=1,
                    marker='o',s=55,label='test_set')
x_combine_scaler = np.vstack((x_train_scaler,x_test_scaler))
y_combine = np.hstack((y_train,y_test))
plot_decision_regions(x_combine_scaler,y_combine,ppn,
                        test_idx=range(105,150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc = 'upper left')
plt.show()
