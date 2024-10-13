if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    
import numpy as np
import dezero
import matplotlib.pyplot as plt

train_set = dezero.datasets.SinCurve(train=True)
print(len(train_set))
print(train_set[0])
print(train_set[1])
print(train_set[2])

# シーケンスデータの読み込み
xs = [example[0] for example in train_set] # あるデータ
ts = [example[1] for example in train_set] # あるデータよりも 1 ステップ先のデータ

plt.plot(np.arange(len(xs)), xs, label='xs')
plt.plot(np.arange(len(ts)), ts, label='ts')
plt.savefig('sin_curve.png')
# plt.show()

