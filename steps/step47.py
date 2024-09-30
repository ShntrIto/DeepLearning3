if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import dezero.functions as F
from dezero.models import MLP
from dezero import Variable, as_variable

def softmax1d(x):
    # より良い実装のためには，x の中の最大値を見つけ，
    # 各要素に対して最大値を引けばよい
    x = as_variable(x)
    y = F.exp(x)
    sum_y = F.sum(y)
    return y/sum_y

model = MLP((10, 3))

x = np.array([[0.2, -0.4]])
y = model(x)
p = softmax1d(y)
print(y)
print(p)