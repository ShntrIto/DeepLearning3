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

# x = np.array([[0.2, -0.4]])
# y = model(x)
# p = softmax1d(y)
# print(y)
# print(p)

x = np.array([[0.2, -0.4], [0.3, 0.5], [1.3, -3.2], [2.1, 0.3]])
t = np.array([2, 0, 1, 0]) # 正解となるクラス番号が教師データとして記録されている（だから，t.data で要素にアクセスできる）

y = model(x)
loss = F.softmax_cross_entropy_simple(y, t)
print(loss)