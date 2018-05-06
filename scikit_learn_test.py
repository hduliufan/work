#数据集scikitlearn自带的iris数据
from sklearn import datasets
import numpy as np
iris=datasets.load_iris()
x = iris.data[:, [2,3]]
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

import matplotlib.pyplot as plt
