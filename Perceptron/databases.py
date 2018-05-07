#product  databases
import pandas as pd
import csv
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
#print(matix1.shape)
dic={'0':a,'1':b,'2':c,'3':d,'4':q}
df=pd.DataFrame(dic)
df.to_csv('trainingdatabases.csv',index=False)
df0=pd.read_csv('trainingdatabases.csv')
print(np.shape(df0))
#print(df)
zipfile = zip (a,b,c,d,q)
df2 = pd.DataFrame(list(zipfile),columns=list('abcde'))
df2.to_csv('trainingdatabases2.csv',index=False)
df3=pd.read_csv('trainingdatabases2.csv')
print(df3.shape)
csvfile=open("trainingdatabases1.csv",'w',newline='')
writer=csv.writer(csvfile)
writer.writerow(list('01234'))
for ai,bi,ci,di,qi in zip(a,b,c,d,q):
    writer.writerow([ai,bi,ci,di,qi])
csvfile.close()
