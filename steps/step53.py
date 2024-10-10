if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import dezero
import numpy as np
import matplotlib.pyplot as plt
from dezero.models import MLP
from dezero import optimizers
import dezero.functions as F
 
def f(x):
    x = x.flatten()
    x = x.astype(np.float32)
    x /= 255.0
    return x

max_epoch = 5
batch_size = 100
hidden_size = 1000

train_set = dezero.datasets.MNIST(train=True, transform=f)
test_set = dezero.datasets.MNIST(train=False, transform=f)
train_loader = dezero.DataLoader(train_set, batch_size)
test_loader = dezero.DataLoader(test_set, batch_size, shuffle=False)

model = MLP((hidden_size, hidden_size, 10), activation=F.relu)
optimizer = optimizers.Adam().setup(model)

# if dezero.cuda.gpu_enable:
#     print('Use GPU!')
#     train_loader.to_gpu()
#     model.to_gpu()

checkpoint_path = 'checkpoint/my_mlp.npz'

if os.path.exists(checkpoint_path):
    print('Load checkpoint file...')
    model.load_weights(checkpoint_path)
    # TODO: なぜ，重みを読み込む場合は model を gpu へ移動させる必要がないの？
else:
    print('Use GPU!')
    train_loader.to_gpu()
    model.to_gpu()

print('Learning Start!')
for epoch in range(max_epoch):
    sum_loss, sum_acc = 0, 0
    for x, t in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        acc = F.accuracy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(t)
        sum_acc += float(acc.data) * len(t)
    
    # 1 エポック終了後に表示
    print('epoch: {}'.format(epoch+1))
    print('train loss: {:.4f}, accuracy: {:.4f}'.format(sum_loss / len(train_set), sum_acc / len(train_set)))
    
model.save_weights(checkpoint_path)