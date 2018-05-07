#stochastic gradient descent
#同步测试
import numpy as np
import random
class AdalineSGD(object):
    '''
    eta learning rata
    n_iter times
    shuffle stochastic
    '''
    def __init__(self,eta,n_iter,shuffle=True,random_state=None):
        self.eta=eta
        self.n_iter=n_iter
        self.w_initialized=False
        self.shuffle=shuffle
        if random_state:
            random.seed(random_state)
    def fit(self,x,y):
        '''
        x=ndarray(n_samples,n_features),training data
        y=ndarray(n_samples),labels一维数组
        returns
        self:object
        w_:1darray,weights after fitting
        '''
        self._initialized_weight(x.shape[1])
        self.cost_ = []
        for _ in range(self.n_iter):
            if self.shuffle:
                x , y = self._shuffle(x , y)
            cost = []
            for xi , yi in zip (x,y):
                cost.append(self._updata_weight(xi,yi))
            avg_cost = sum(cost)/len(y)
            self.cost_.append(avg_cost)
        return self
    '''
    处理流数据的在线学习
    '''
    def partial_fit(self,x,y):
        if not self.w_initialized:
            self._initialized_weight(x.shape[1])
        #ravel is successiful?
        if y.ravel().shape[0] > 1:
            for xi ,yi in zip(x , y):
                self._updata_weight(xi ,yi)
        else:
            self._updata_weight(x,y)
        return self
    def _initialized_weight(self,m):
        '''
        initialized weights
        '''
        self.w_ = np.zeros(m+1)
        self.w_initialized = True
    def _updata_weight(self,xi,yi):
        output=self.net_input(xi)
        errors=yi-output
        self.w_[1:] = self.eta*xi.dot(errors)
        self.w_[0]  = self.eta*errors
        cost = 0.5*errors**2
        return cost
    def _shuffle(self,x,y):
        r = np.random.permutation(len(y))
        return x[r] , y[r]
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
