if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    
from dezero.dataloaders import SeqDataLoader    
import numpy as np
import dezero
import matplotlib.pyplot as plt
import dezero.functions as F
from dezero.models import Model
import dezero.layers as L

# train_set = dezero.datasets.SinCurve(train=True)
# dataloader = SeqDataLoader(train_set, batch_size=3)
# x, t = next(dataloader)
# print(x)
# print('-----')
# print(t)

max_epoch = 100
batch_size = 30
hidden_size = 100
bptt_length = 30

train_set = dezero.datasets.SinCurve(train=True)
dataloader = SeqDataLoader(train_set, batch_size=batch_size)
seqlen = len(train_set) # bptt_length で割り切れないことがあるため

class BetterRNN(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.rnn = L.LSTM(hidden_size)
        self.fc = L.Linear(out_size)
        
    def reset_state(self):
        self.rnn.reset_state()
    
    def forward(self, x):
        h = self.rnn(x)
        y = self.fc(h)
        return y

model = BetterRNN(hidden_size, 1)
optimizer = dezero.optimizers.Adam().setup(model)

for epoch in range(max_epoch):
    model.reset_state()
    loss, count = 0, 0
    
    for x, t in dataloader:
        y = model(x)
        loss += F.mse(y, t)
        count += 1
        
        if count % bptt_length == 0 or count == seqlen:
            dezero.utils.plot_dot_graph(loss, to_file='lstm.pdf')
            model.cleargrads()
            loss.backward()
            loss.unchain_backward()
            optimizer.update()
    
    avg_loss = float(loss.data) / count
    print('| epoch %d | loss %f' % (epoch + 1, avg_loss))
    

xs = np.cos(np.linspace(0, 4 * np.pi, 1000))
model.reset_state()
pred_list = []

with dezero.no_grad():
    for x in xs:
        x = np.array(x).reshape(1, 1)
        y = model(x)
        pred_list.append(float(y.data))

plt.plot(np.arange(len(xs)), xs, label='y=cos(x)')
plt.plot(np.arange(len(xs)), pred_list, label='predict')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.savefig('lstm_sin-graph.pdf')