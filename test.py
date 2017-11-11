import numpy as np
from numpy.linalg import inv

x = np.array([ [[ 1., 0.],[0., 1.]]])
print(x)
print(x.shape)
y = np.repeat(x,10,axis=0)
print(y)
print(y.shape)
y[2]=[[43.,65.],[43.,8.]]
print(y)
print(y[2])
print('y[0]',y[0])
print('inv(y[0])',inv(y[0]))
