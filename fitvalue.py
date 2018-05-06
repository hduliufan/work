#fitvalue
import numpy as np
#x1x2 is ndarray (n*1)
#a empty ndarray to save fitvalue (n*1)
def fitvalue (x1,x2):
    fitvalues=np.zeros((np.shape(x1)[0],1))
    i=0
    for x_1,x_2 in zip(x1,x2):
        fitvalues[i,:]=21.5+x_1*np.sin(4*np.pi*x_1)+x_2*np.sin(20*np.pi*x_2)
        i=i+1
    return fitvalues
