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
