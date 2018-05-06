#变异算子采用标准正太分布

import numpy as np
import random
def variation(N,x1,x2):
    normal=np.random.normal(loc=0,scale=1.0,size=[N,2])
    for i in range(N):
        x1[i]=x1[i]+normal[i,0]
        x2[i]=x2[i]+normal[i,0]
    return x1,x2
