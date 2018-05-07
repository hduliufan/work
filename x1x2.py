#种群解码即二进制转换为十进制
#每个父本由t1t2组成
#t=t1+t2即父本个体长度
import numpy as np
import math
#定义空数组存放x1x2
#种群规模：pop，个体父本,x1,x2长度t,t1,t2,
def x1_x2(pop,t,t1,t2,a1,b1,a2,b2):
    x1=np.zeros((np.shape(pop)[0],1))
    x2=np.zeros((np.shape(pop)[0],1))
    for i in range(np.shape(pop)[0]):
        x1[i,:]=a1+twototen(pop[i,:t1],t1)*(b1-a1)/(math.pow(2,t1)-1)
        x2[i,:]=a2+twototen(pop[i,t1:],t2)*(b2-a2)/(math.pow(2,t2)-1)
    return x1,x2
#二进制变为十进制p为数组1，t为长度
def twototen(p,t):
    sum=0
    for i in range(t):
        sum=sum+p[i]*math.pow(2,t-1-i)
    return sum
