if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dezero import Model
import numpy as np
import dezero.layers as L
import dezero.functions as F

# rnn = L.RNN(10)
# x = np.random.rand(1, 1) # (1, 1) サイズの入力データを作成
# h = rnn(x)
# print(h.shape)

class SimpleRNN(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.rnn = L.RNN(hidden_size)
        self.fc = L.Linear(out_size) # 隠れ状態を受け取って，出力を作るために必要
        
    def reset_state(self):
        self.rnn.reset_state()
    
    def forward(self, x):
        h = self.rnn(x)
        y = self.fc(h)
        return y
    
seq_data = [np.random.rand(1, 1) for _ in range(1000)] # ダミーのシーケンスデータ
xs = seq_data[0:-1]
ts = seq_data[1:]

model = SimpleRNN(10, 1)

loss, cnt = 0, 0
for x, t in zip(xs, ts):
    y = model(x)
    loss += F.mse(y, t)
    
    cnt += 1
    if cnt == 2:
        model.cleargrads()
        loss.backward()
        break
    
model.plot(xs[0], to_file='simple_rnn-x0.png')
model.plot(xs[1], to_file='simple_rnn-x1.png')