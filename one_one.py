import numpy as np
import matplotlib.pyplot as plt
from getlen_bit import getlen
from begain import getbegain
from x1x2 import x1_x2
from exchange_normal import variation
from fitvalue import fitvalue
#计算精度（人为设定）0.001
s=0.0001
a1=-3
a2=4.1
b1=12.1
b2=5.8
#种群规模
N=20
#-3<=x1<=12.1   4.1<=x2<=5.8
#二进制长度t1,t2
t1=getlen(a1,b1,s)
t2=getlen(a2,b2,s)
#print(t1,t2)
t=t1+t2
#print(t)
#二进制种群（N*t）
pop=getbegain(N,t)
#print(pop)
x1,x2=x1_x2(pop,t,t1,t2,a1,b1,a2,b2)
#print(x1,x2)

def one_one(x1,x2,N):
    T=0
    #记录变异前的最大适应值
    fit1_=[]
    #记录变异后的最大适应值
    fit2_=[]
    #记录最终的适应值
    fit=[]
    while(T<10):
        #父本个体适应值（N*1）
        fit1=fitvalue(x1,x2)
        fit1_.append(np.max(fit1))
        #变异采用高斯算子即N（0，1）标准正太分布
        x11,x22=variation(N,x1,x2)
        #变异后个体适应值（N*1）
        fit2=fitvalue(x11,x22)
        fit2_.append(np.max(fit2))
        #记录索引
        i=0
        for fit_1,fit_2 in zip(fit1,fit2):
            if fit_1<fit_2:
                #变异前与变异后交换
                x1[i]=x11[i]
                x2[i]=x22[i]
            i=i+1
        T=T+1
    #输出最大适应值变化
    for b,a in zip (fit1_,fit2_):
        fit.append(np.where(b>a,b,a))
    #输出图形变异前后的最大适应值变化
    plt.title('1+1')
    plt.subplot(111)
    plt.xlabel('T')
    plt.ylabel("fitvalue")
    plt.plot(range(1,T+1),fit1_,'b--',label='bef')
    plt.plot(range(1,T+1),fit2_,'g--',label='aft')
    plt.legend(loc='upper left')
    plt.plot(range(1,T+1),fit,'r-',label='maxfit')
    plt.show()
one_one(x1,x2,N)
#print(x1,x2)
