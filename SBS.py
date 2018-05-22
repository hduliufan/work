#序列反向选择算法sbs
from sklearn.base import clone
#itertools迭代器产生
from itertools import combinations
from sklearn.metrics  import accuracy_score
import numpy as np 
from sklearn.cross_validation import train_test_split
class SBS(object):
    '''
    estimator 是采用的方法分类后的模型
    '''
    def __init__(self,estimator, k_feature, scoring=accuracy_score,
    test_size= None ,random_state=None):
        self.estimator= clone(estimator)
        self.k_feature= k_feature
        self.scoring= scoring
        self.random_state = random_state
        self.test_size= test_size
    def fit(self,x,y):
        x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=self.test_size,
                                                        random_state=self.random_state)
        dim=x_train.shape[1]
       #indices 目录 元组不能改变
        #类内全局变量
        self.indices_= tuple(range(dim))
        #子集subset
        self.subsets_=[self.indices_]
        score= self._calc_score(x_train,y_train,x_test,y_test,self.indices_)
        #数组的一个元素
        self.scores_= [score]

        while dim > self.k_feature:
            scores=[]
            subsets= []
            for p in combinations(self.indices_, r=dim-1):
                score= self._calc_score(x_train,y_train,x_test,y_test,p)
                scores.append(score)
                #子集存储
                subsets.append(p)
        #argmax返回值是最大值的indices
            best= np.argmax(score)
            #返回的是最优的子集即score最大的子集目录即列向量标号
            self.indices_= subsets[best]
            self.subsets_.append(self.indices_)
            dim -=1
            #存储的是score最大的子集
            self.scores_.append(scores[best])
        #返回的是满足阈值的最佳score
        self.k_scores_=self.scores_[-1]
        return self
    #返回最佳的特征列
    def transform(self,x):
        return x[:,self.indices_]
    def _calc_score(self,x_train,y_train,x_test,y_test,indices):
        self.estimator.fit(x_train[:,indices],y_train)
        y_predict= self.estimator.predict(x_test[:,indices])
        #实际是调用accuracy_score 进行正确率
        score= self.scoring(y_test,y_predict)
        return score
    def bestchoice(self):
        best= np.argmax(self.scores_)
        return self.subsets_[best]
