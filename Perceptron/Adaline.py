#BGD
'''
batch gradient descent
'''
import numpy as np
class Adaline(object):
    #eta learning rata
    #n_iter times
    def __init__(self,eta,n_iter):
        self.eta=eta
        self.n_iter=n_iter
    def fit(self,x,y):
        '''
        x=ndarray(n_samples,n_features),training data
        y=ndarray(n_samples),labels
        returns
        self:object
        w_:1darray,weights after fitting
        errors=list,errors times
        '''
        #init
        self.w_=np.zeros(np.shape(x)[1]+1)
        self.cost_=[]
        for _ in range(self.n_iter):
            #ndarray
            output=self.net_input(x)
            #ndarray
            errors=y-output
            self.w_[1:] += self.eta*x.T.dot(errors)
            self.w_[0] += self.eta*errors.sum()
            cost = 1/2*((errors**2).sum())
            self.cost_.append(cost)
        return self

    def net_input(self,x):
        '''
        calculate net input
        '''
        return np.dot(x,self.w_[1:])+self.w_[0]
    def predict(self,x):
        '''
        positive function
        '''
        return np.where(self.net_input(x)>=0.0,1,-1)
#painting
import matplotlib.pyplot as plt
#from perception import perceptron
#read data as DaraFrame
import pandas as pd
import numpy as np
import os
import pandas as pd
import numpy as np
import random
path=os.getcwd()+'\\trainingdatabases0.csv'
df=pd.read_csv(path,header=0,names=list('01234'))
y=df.iloc[0:100,4].values
y=np.where(y==0,-1,1)
x=df.iloc[0:100,[0,2]].values
'''
标准差归一化
'''
'''
也可极差归一化
'''
x[:,0]=(x[:,0]-x[:,0].mean())/x[:,0].std()
x[:,1]=(x[:,1]-x[:,1].mean())/x[:,1].std()
plt.scatter(x[:50,0],x[:50,1],color='red',marker='o',label='setosa')
plt.scatter(x[50:100,0],x[50:100,1],color='green',marker='x',label='versicolor')
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc='upper right')
plt.show()
ppn=Adaline(eta=0.01,n_iter=15)
ppn.fit(x,y)
plt.plot(range(1,len(ppn.cost_)+1),np.log10(ppn.cost_),marker='o',color='red')
plt.xlabel('epochs')
plt.ylabel('sum-squared-errors')
plt.show()
