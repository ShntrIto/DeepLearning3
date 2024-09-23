if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable
import dezero.functions as F 

x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
y = F.reshape(x, (6,))
y.backward(retain_grad=True)
print(x.grad)

# reshape の確認
p = Variable(np.array([[1,2,3], [4,5,6]]))
q = p.reshape((2, 3))
r = p.reshape(2, 3)

# transpose の確認
p = Variable(np.random.rand(1,2,3,4))
print(p.shape)
p = p.transpose(2,3,0,1)
print(p.shape)