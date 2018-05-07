import numpy as np
x=np.zeros(5)
j=[[1,2,3,4]]
j=np.array(j)
print(type(j))
print(np.shape(j))
k=[[1,2,3,4],[4,5,6,8]]
k=np.array(k)
print(type(k))
print(np.shape(k))
q=np.dot(k,j.T)

print(q)
