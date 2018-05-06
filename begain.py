#初始种群的产生种群数目为N
#二进制产生二进制长度为即位数为t
#随机产生二进制
import random
import numpy as np
#产生初始种群空数组
def getbegain(N,t):
    chromosomes=np.zeros((N,t))
    np.random.seed(100)
    for i in range (N):
        chromosomes[i,:]=np.random.randint(0,2,t)
    return chromosomes
#a=getbegain(20,33)
#print(a)
