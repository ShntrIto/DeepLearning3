if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import dezero.functions as F
from dezero.datasets import Spiral
from dezero import DataLoader
from dezero import optimizers
from dezero.models import MLP

batch_size = 30
max_epoch = 300
hidden_size = 10
lr = 1.0

train_set = Spiral(train=True)
test_set = Spiral(train=False)
train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size, shuffle=False)

model = MLP((hidden_size, 10))
optimizer = optimizers.SGD(lr).setup(model)

import matplotlib.pyplot as plt

train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

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

    train_loss = sum_loss / len(train_set)
    train_acc = sum_acc / len(train_set)
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    print('epoch: {}'.format(epoch+1))
    print('train loss: {:.4f}, accuracy: {:.4f}'.format(train_loss, train_acc))

    sum_loss, sum_acc = 0, 0
    with dezero.no_grad():
        for x, t in test_loader:
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            acc = F.accuracy(y, t)
            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)

    test_loss = sum_loss / len(test_set)
    test_acc = sum_acc / len(test_set)
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)

    print('test loss: {:.4f}, accuracy: {:.4f}'.format(test_loss, test_acc))

# Plotting the results
epochs = range(1, max_epoch + 1)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label='Train loss')
plt.plot(epochs, test_losses, label='Test loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss over epochs')

plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, label='Train accuracy')
plt.plot(epochs, test_accuracies, label='Test accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy over epochs')

plt.tight_layout()
plt.savefig('training_results.png')  # Save the figure as an image file
plt.show()

    