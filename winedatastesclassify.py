#降维（对特征缩减）
# 数据特征处理sbs后向选择
# random forest classifier 随机森林选择forest_importances_
import pandas as pd 
from sklearn.cross_validation import train_test_split
import numpy as np 
import random 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
#读取数据
name = ['classlabel', 'alcohol', 'malic acid', 'ash', 'alcalinity of ash',
        'magnesium', 'total phenols', 'flavanoids', 'nonflavanoids phenols',
        'proanthocyanins', 'color intensity', 'hue', 'od280/od315 of dilute',
        'proline']
df= pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',
                 header=0,names=name)
print('class labels', np.unique(df['classlabel']))
'''
保存数据为csv文件
csv=pd.DataFrame(df)
csv.to_csv('winedatabases.csv',index=False , sep=',')


print(df.shape)
classlabel={1:'hello',2:'good',3:'asdf'}
df['classlabel']=df['classlabel'].map(classlabel)

from sklearn.preprocessing import LabelEncoder
df1=LabelEncoder()
#输出为数组
df=df1.fit_transform(df['classlabel'])
'''
#train 和 test集
from sklearn.cross_validation import train_test_split
x,y= df.iloc[:,1:].values, df.iloc[:,0].values
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.3,
                                                    random_state= 0)


#标准化
from sklearn.preprocessing import StandardScaler
std= StandardScaler()
x_train_std= std.fit_transform(x_train)
x_test_std= std.transform(x_test)

#选择特征进行降维SBS序列后向选择
from SBS import *


#采用svm中核rbf
from sklearn.svm import SVC
svm= SVC(kernel='rbf',gamma=1.0,C=1000.0,random_state=0)
#gamma是决策边界的参数gamma小决策边界宽松
#C是正则化系数倒数C越大对惩罚要求高
sbs=SBS(svm,k_feature=1,test_size=0.2,random_state=0)
sbs.fit(x_train_std,y_train)
#取最佳的特征标号
feature=sbs.bestchoice()

#采用KNN
from sklearn.neighbors import KNeighborsClassifier
knn= KNeighborsClassifier(n_neighbors=2)
sbs2=SBS(knn,k_feature=1,test_size=0.2,random_state=0)
sbs2.fit(x_train_std,y_train)
feature2=sbs2.bestchoice()


'''
输出最佳特征数量与准确率图型
train data
'''

ks_feature= [len(i) for i in sbs.subsets_]
k_feature= [len(j) for j in sbs2.subsets_]
plt.plot(ks_feature,sbs.scores_,'r',marker='x',label='svm')
plt.plot(k_feature,sbs2.scores_,'g',marker='s',label='knn')
plt.grid(True)
plt.legend()
plt.xlabel('feature num')
plt.ylabel('training accuracy')
plt.show()

print(sbs.subsets_)

print('feature index',sbs.subsets_[len(name)-len(feature)])
print(feature)

for i in feature:

        print(name[i+1])
svm.fit(x_train_std[:,feature],y_train)
knn.fit(x_train_std[:, feature2], y_train)
#过拟合

print('svmtest accuracy',svm.score(x_test_std[:,feature],y_test))
print('svmtrain accuracy',svm.score(x_train_std[:,feature],y_train))
print('knntrain accuracy', knn.score(x_train_std[:,feature2],y_train))
print('knntest accuracy', knn.score(x_train_std[:, feature2], y_train))
print(svm.intercept_)

#.intercept_截距项
from sklearn.linear_model import LogisticRegression
#构造库

#l1正则化是产生稀疏矩阵norm(w,1)列和范数
#l2正则化是二次即norm（w，2）谱范数
lr= LogisticRegression(penalty='l1')
#调用库
lr.fit(x_train_std,y_train)
#测试集预测#score= sklearn.metrics.accuracy_score显示正确分类比例
print(lr.score(x_train_std,y_train))
print(lr.score(x_test_std,y_test))
#未出现过拟合
#.intercept_是截距项logistic regression默认是OVR即截距项为1（2，3）；2（1，3）；
# 3(1,2)
print(lr.intercept_)
#.coef_是输出权重系数数组，每一个向量为权重值向量即w1,w2,w3.....
print(lr.coef_)

'''
from SVMnon_linear import plot_decision_regions
plt.figure(1)
plt.subplot(111)

#(二特征)画散点图scatter
#降维？
plt.figure(2)

plt.scatter(x_train_std[y_train== 1,0],x_train_std[y_train== 1,1],marker='x',c='r',label='1')
plt.scatter(x_train_std[y_train== 2,0],x_train_std[y_train==2,1],c='g',marker= 'o',label='2')
plt.show()
'''
#随机森林法randomforestclassifier确定特征的重要程度
from sklearn.ensemble import RandomForestClassifier
feat_labels =df.columns[1:]
#print(feat_labels)
#n_estimator 决策树数量，predict_proba()输出数组预测值概率
forest= RandomForestClassifier(n_estimators=10000,random_state=0,n_jobs=-1)
forest.fit(x_train,y_train)
#feature_importances_评估特征重要程度
importances= forest.feature_importances_
#print(importances)
#argsort()将数组元素由小到大排序并返回排序的索引
indices= np.argsort(importances)[::-1]


plt.figure(2)
plt.subplot(111)
#输出重要程度图bar条形图
#align对齐
plt.bar(range(x_train.shape[1]),importances[indices],
        color= 'lightblue',align='center')
#xticks设置轴标号
#rotation标记旋转角度
plt.xticks(range(x_train.shape[1]),feat_labels[indices],rotation=90)
plt.show()



