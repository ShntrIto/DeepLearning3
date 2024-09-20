if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable

def f(x):
    y = x ** 4 - 2 * x** 2
    return y

x = Variable(np.array(2.0))
iters = 10

# ニュートン法による最適化
for i in range(iters):
    print(i, x)
    y = f(x)
    x.cleargrad()
    y.backward(create_graph=True)

    gx = x.grad
    # 勾配をリセットする
    # 計算の途中の変数の勾配はデフォルトでリセットされるが，
    # 末端の変数だけはリセットされないため，cleargrad()が必要
    x.cleargrad()
    gx.backward()
    gx2 = x.grad

    x.data -= gx.data / gx2.data