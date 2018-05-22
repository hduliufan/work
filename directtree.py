#决策数（二叉树）
'''
属于同一种类别衡量标准不同的影响
'''
#三种决策不纯度衡量标准
#Gini index 基尼系数 Ig（p）= p*(1-p)+(1-p)*p
#entropy 熵 IH(p)=-(plog2(p)+ (1-p)log2(1-p))
#classification error IE（p）= 1 - max(p,1-p)
#熵和基尼系数不纯度衡量是
import  numpy  as np 
import matplotlib.pyplot as plt
def gini( p):
    return p* (1- p)+ (1- p)* (1-(1- p))

def entropy(p):
    return -p* np.log2(p)- (1- p )* np.log2(1- p)
def classification_error(p):
    return 1- np.max([p,1- p])

x=np.arange(0.0,1.0,0.01)
entr= [entropy(p) if p!= 0 else 0 for p in x]

        #对熵进行缩放0.5倍即entropy/2
sc_entr= [e*0.5 if e else None for e in entr]
err= [classification_error(i) for i in x]
print(entr)
plt.figure(1)
plt.subplot(111)
for i, labe, mar, c in zip ([entr, sc_entr, gini(x),err ],
                                    ['entropy','sc_entr','gini','classification_error'],
                                    ['-','-.','--','-'],
                                    ['r','k','b','g']):

    plt.plot(x,i,label=labe,linestyle=mar,color=c)                            
plt.xlabel('i=1')
plt.ylim([0,1.1])
plt.legend(loc='upper left')    


 
plt.show()