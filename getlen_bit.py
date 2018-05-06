#确定二进制长度t
import numpy as np
import math
def getlen(a,b,s):
    t=0
    isfalse=False
    while(not isfalse):
        if (math.pow(2,t)-1<(b-a)/s):
            t=t+1
        else:
            isfalse=True
    return t
