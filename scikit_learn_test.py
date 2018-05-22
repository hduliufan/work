#数据集scikitlearn自带的iris数据
from sklearn import datasets
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier


#plot_decision_regions画图
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
    '''
    scatter(x,y,color,alpha()透明度,linewidth(线宽),marker,s(散点大小))
    '''
    if test_idx:
        x_test,y_test=x[test_idx,:],y[test_idx]
        plt.scatter(x_test[:,0],x_test[:,1],
                    c='k',alpha=1.0,linewidths=1,
                    marker='o',s=5,label='test_set')

iris=datasets.load_iris()
x = iris.data[:, [2,3]]
#一维数组
y = iris.target
#数据集分类，训练集train-data，测试集test-data
#cross_validation数据分类库

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=0)
#数据标准化standardized
#训练集和测试集有同样的均值和方差
#preprocessing 数据预处理库

scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaler = scaler.transform(x_train)
x_test_scaler = scaler.transform(x_test)



ppn = Perceptron(eta0=0.01,random_state=0,n_iter=40)
ppn.fit(x_train_scaler,y_train)

y_predict = ppn.predict(x_test_scaler)
print('misclassified samples:%d'%(y_predict!=y_test).sum())
print('classified rata:%.2f'%(1-(y_predict!=y_test).sum()/len(y_test)))

#metrics正确率库accuracy_score

print('classified rata:%.2f'%accuracy_score(y_test,y_predict))




x_combine_scaler = np.vstack((x_train_scaler,x_test_scaler))
y_combine = np.hstack((y_train,y_test))


plt.title('ppn')
plot_decision_regions(x_combine_scaler,y_combine,ppn,
                        test_idx=range(105,150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc = 'upper left')


#sklearn.svm .SVC

svm = SVC(kernel='rbf', gamma= 100.0, C=10.0,random_state=0)#C为正则系数的倒数c越大则惩罚越大
#几何间隔小，c越小，惩罚不严格，几何间隔越大
svm.fit(x_train_scaler , y_train)
y_predict_3 = svm.predict(x_test_scaler)
print('svmrbf classified rata:%.2f'%(1 - (y_predict_3 != y_test).sum() / len(x_test_scaler)))
plt.figure(5)
plt.subplot(111)
plt.title('svm_rbf')
plot_decision_regions(x_combine_scaler,y_combine,
                        classifier=svm,
                        test_idx=range(105,150),
                        resolution=0.02)
plt.xlabel('花瓣长度')
plt.ylabel('花瓣宽度')
plt.legend(loc='upper left')


svm = SVC(kernel='linear', C=1.0, random_state=0)#C为正则系数的倒数c越大则惩罚越大
#几何间隔小，c越小，惩罚不严格，几何间隔越大
svm.fit(x_train_scaler , y_train)
y_predict_4 = svm.predict(x_test_scaler)
print('svmlinear classified rata:%.2f'%(1 - (y_predict_4 != y_test).sum() / len(x_test_scaler)))
plt.figure(6)
plt.subplot(111)
plt.title('svm_linear')
plot_decision_regions(x_combine_scaler,y_combine,
                        classifier=svm,
                        test_idx=range(105,150),
                        resolution=0.02)
plt.xlabel('花瓣lenth')
plt.ylabel('花瓣width')
plt.legend(loc='upper left')


#logisticregression

lr = LogisticRegression(C = 1000.0,random_state = 0)
lr.fit (x_train_scaler, y_train)
y_predict_1 = lr.predict(x_test_scaler)
print('logic classified rata:%.2f'%(1 - (y_predict_1 != y_test).sum() / len(x_test_scaler)))

plt.figure(2)
plt.subplot(111)
plt.title('LogisticRegression')
plot_decision_regions(x_combine_scaler,
                        y_combine,
                        classifier = lr,
                        test_idx = range(105,150))

plt.xlabel(' pl')
plt.ylabel('pw')
plt.legend( loc = 'upper left')


#SGDClassifier

ppn = SGDClassifier( loss = 'perceptron')
lr = SGDClassifier(loss = 'log')
svm = SGDClassifier(loss = 'hinge')
lr.fit (x_train_scaler, y_train)
y_predict_2 = lr.predict(x_test_scaler)
print('sgdclassified rata:%.2f'%accuracy_score(y_test,y_predict_2))
plt.figure(3)
plt.subplot(111)
plt.title('SGDClassifier')
plot_decision_regions(x_combine_scaler,
                        y_combine,
                        classifier = lr,
                        test_idx = range(105,150))

plt.xlabel(' pl')
plt.ylabel('pw')
plt.legend( loc = 'upper left')





plt.show()
