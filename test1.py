import numpy as np
print(np.ones([2,5]))
a =np.array([ [1,2,3,4,5]]).T
b =np.array([ [6,7,8,9,10]]).T
print(a.shape)
print(np.hstack((a,b)))