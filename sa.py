import numpy as np
a=[1.0,2.0,3.0]
a=np.matrix(a)
b=[1.0,2.0,3.0]
b=np.matrix(b)
print(np.dot(a,b.T))
