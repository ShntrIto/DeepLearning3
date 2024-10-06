if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import dezero
import matplotlib.pyplot as plt

train_set = dezero.datasets.MNIST(train=True, transform=None)
test_set = dezero.datasets.MNIST(train=False, transform=None)

print(len(train_set))
print(len(test_set))

x, t = train_set[0]
# X11 fowarding を使って VSCode と Xming で表示しようと思ったけど
# 上手くいかなかったので後回し
# plt.imshow(x.reshape(28, 28), cmap='gray') 
# plt.axis('off')
plt.imsave('./outputs/mnist_sample.png', x.reshape(28, 28), cmap='gray')
print('label:', t)
print(type(x), x.shape)
print(t)