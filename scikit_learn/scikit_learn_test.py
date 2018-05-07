#scikit-learn
from sklearn import datasets
import pandas as pd
import csv
import numpy as np
iris=datasets.load_iris()
x=iris.data[: , :]
y=iris.target
'''
产生鸢尾花数据csv文件
csvfile = open('trainingdatabases0.csv','w',newline='')
csvweiter = csv.writer(csvfile)
csvweiter.writerow(list('01234'))
for index in range(x.shape[0]):
    csvweiter.writerow([x[index,0],x[index,1],x[index,2],x[index,3],y[index]])
csvfile.close()
df = pd.read_csv('trainingdatabases0.csv',header=0)

print(df.tail())
print(type(x))
print(y)
'''
