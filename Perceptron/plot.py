import numpy as np
class perceptron(object):
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
        self.errors_=[]
        for _ in range(self.n_iter):
            errors=0
            for xi,yi in zip(x,y):
                updata=self.eta*(self.predict(xi)-yi)
                self.w_[1:]+=updata*xi
                self.w_[0]+=updata
                errors+=int(updata!=0.0)
            self.errors_.append(errors)
        print(self.errors_)
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
a=np.random.uniform(6.0,7.0,150)
b=np.random.uniform(2.0,4.0,150)
c=np.random.uniform(5.0,5.5,150)
d=np.random.uniform(1.5,2.5,150)
q=[]
for i in range(150):
    e=np.random.choice(['a','b'])
    q.append(e)
dic={'0':a,'1':b,'2':c,'3':d,'4':q}
df=pd.DataFrame(dic)
y=df.iloc[0:100,4].values
y=np.where(y=='b',-1,1)
x=df.iloc[0:100,[0,2]].values
plt.scatter(x[:50,0],x[:50,1],color='red',marker='o',label='setosa')
plt.scatter(x[50:100,0],x[50:100,1],color='green',marker='x',label='versicolor')
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc='upper right')
plt.show()
ppn=perceptron(eta=1,n_iter=10000)
ppn.fit(x,y)
plt.plot(range(1,len(ppn.errors_)+1),ppn.errors_,marker='o',color='red')
plt.xlabel('epochs')
plt.ylabel('number of miscalssifications')
plt.show()
