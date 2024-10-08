if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero.core import Function, Variable, as_variable, as_array
from dezero import utils, cuda

class Sin(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.sin(x)
        return y
    
    def backward(self, gy):
        x, = self.inputs
        gx = gy * cos(x)
        return gx

def sin(x):
    return Sin()(x)

class Cos(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.cos(x)
        return y
    
    def backward(self, gy):
        x, = self.inputs
        gx = gy * -sin(x)
        return gx

def cos(x):
    return Cos()(x)

class Tanh(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.tanh(x)
        return y
    
    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * (1 - y * y)
        return gx

def tanh(x):
    return Tanh()(x)

class Exp(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.exp(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]() # weakref 弱参照
        gx = gy * y
        return gx

def exp(x):
    return Exp()(x)

class Log(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.log(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy / x
        return gx

def log(x):
    return Log()(x)

class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape
    
    def forward(self, x):
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return y
    
    def backward(self, gy):
        return reshape(gy, self.x_shape)
    
def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)

class Transpose(Function):
    # 順伝播も逆伝播も，入力（出力）の転置を返すだけで良いので難しいことは考えなくていい
    def __init__(self, axes=None):
        self.axes = axes
    
    def forward(self, x):
        y = x.transpose(self.axes)
        return y
    
    def backward(self, gy):
        if self.axes is None:
            return transpose(gy)

        axes_len = len(self.axes)
        inv_axes = tuple(np.argsort([ax % axes_len for ax in self.axes]))
        return transpose(gy, inv_axes)
    
def transpose(x, axes=None):
    return Transpose(axes)(x)

class Sum(Function):
    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        self.x_shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, gy):
        # ただ，入力と同じ形の勾配にして返すだけ
        gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
        gx = broadcast_to(gy, self.x_shape)
        return gx
    
def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)

class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        xp = cuda.get_array_module(x)
        y = xp.broadcast_to(x, self.shape)
        return y
    
    def backward(self, gy):
        gx = sum_to(gy, self.x_shape)
        return gx
    
def broadcast_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)

class SumTo(Function):
    def __init__(self, shape):
        self.shape = shape
    
    def forward(self, x):
        self.x_shape = x.shape
        y = utils.sum_to(x, self.shape)
        return y

    def backward(self, gy):
        gx = broadcast_to(gy, self.x_shape)
        return gx

def sum_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)

class MatMul(Function):
    def forward(self, x, W):
        y = x.dot(W)
        return y
    
    def backward(self, gy):
        x, W = self.inputs
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW
    
def matmul(x, W):
    return MatMul()(x, W)

class MSE(Function):
    # メモリ効率化のためには，関数のスコープを抜けたら自動的にメモリから消去された方が良い
    # そのため，単なる関数ではなく，Function を継承した MSE インスタンスを使って計算すると良い
    def forward(self, x0, x1):
        diff = x0 - x1
        y = (diff ** 2).sum() / len(diff)
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        diff = x0 - x1
        gy = broadcast_to(gy, diff.shape)
        gx0 = gy * diff * (2. / len(diff))
        gx1 = -gx0
        return gx0, gx1

def mse(x0, x1):
    return MSE()(x0, x1)

class Linear(Function):
    def forward(self, x, W, b):
        y = x.dot(W)
        if b is not None:
            y += b
        return y
    
    def backward(self, gy):
        x, W, b = self.inputs
        gb = None if b.data is None else sum_to(gy, b.shape) # TODO: なぜsum_to(gy, b)なのか分からないから理解する
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW, gb
    
def linear(x, W, b=None):
    return Linear()(x, W, b)

def linear_simple(x, W, b=None):
    # メモリ改善のトリックを使った linear 関数
    # Variable インスタンスの t が必要とされないことを考慮して，計算後に t をメモリから解放する
    x, W = as_variable(x), as_variable(W)
    t = matmul(x, W)
    if b is None:
        return t

    y = t + b
    t.data = None
    return y

class Sigmoid(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.tanh(x * 0.5) * 0.5 + 0.5
        return y

    def backward(self, gy):
        y = self.outputs[0]() # 弱参照
        gx = gy * y * (1-y)
        return gx

def sigmoid(x):
    return Sigmoid()(x)

def sigmoid_simple(x):
    x = as_variable(x)
    y = 1 / (1 + exp(-x))
    return y

class GetItem(Function):
    def __init__(self, slices):
        self.slices = slices
    
    def forward(self, x):
        y = x[self.slices]
        return y

    def backward(self, gy):
        x, = self.inputs
        f = GetItemGrad(self.slices, x.shape)
        return f(gy)

def get_item(x, slices):
    return GetItem(slices)(x)

class GetItemGrad(Function):
    def __init__(self, slices, in_shape):
        self.slices = slices
        self.in_shape = in_shape
    
    def forward(self, gy):
        xp = cuda.get_array_module(gy)
        gx = xp.zeros(self.in_shape)
        xp.add.at(gx, self.slices, gy) # gx の self.slices に gy を加算する
        return gx
    
    def backward(self, ggx):
        return get_item(ggx, self.slices)

def softmax_simple(x, axis=1):
    x = as_variable(x)
    y = exp(x)
    sum_y = sum(y, axis=axis, keepdims=True)
    return y / sum_y

class Softmax(Function):
    # DeZero からそのまま引用
    # 自分の実装は何かが間違っている...
    def __init__(self, axis=1):
        self.axis = axis

    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = x - x.max(axis=self.axis, keepdims=True)
        y = xp.exp(y)
        y /= y.sum(axis=self.axis, keepdims=True)
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = y * gy
        sumdx = gx.sum(axis=self.axis, keepdims=True)
        gx -= y * sumdx
        return gx
    
# class Softmax(Function):
#     def __init__(self, axis=1):
#         self.axis = axis
    
#     def forward(self, x):
#         x = as_variable(x)
#         y = exp(x)
#         sum_y = sum(y, axis=self.axis, keepdims=True)
#         return y / sum_y

#     def backward(self, gy):
#         # DeZero からそのまま引用
#         # backward の導出が理解できていない
#         y = self.outputs[0]()
#         gx = y * gy
#         sumdx = gx.sum(axis=self.axis, keepdims=True)
#         gx -= y * sumdx
#         return gx

def softmax(x, axis=1):
    return Softmax(axis)(x)

class Clip(Function):
    def __init__(self, x_min, x_max):
        self.x_min = x_min
        self.x_max = x_max
    
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.clip(x, self.x_min, self.x_max)
        return y
    
    def backward(self, gy):
        x, = self.inputs
        mask = (x.data >= self.x_min) * (x.data <= self.x_max)
        gx = gy * mask
        return gx
    
def clip(x, x_min, x_max):
    return Clip(x_min, x_max)(x)

class SoftmaxCrossEntropy(Function):
    # DeZero からそのまま引用
    # forward は理解したが，backward は理解してない
    def forward(self, x, t):
        N = x.shape[0]
        log_z = utils.logsumexp(x, axis=1)
        log_p = x - log_z
        log_p = log_p[np.arange(N), t.ravel()]
        y = -log_p.sum() / np.float32(N)
        return y

    def backward(self, gy):
        x, t = self.inputs
        N, CLS_NUM = x.shape

        gy *= 1/N
        y = softmax(x)
        # convert to one-hot
        xp = cuda.get_array_module(t.data)
        t_onehot = xp.eye(CLS_NUM, dtype=t.dtype)[t.data]
        y = (y - t_onehot) * gy
        return y

def softmax_cross_entropy(x, t):
    return SoftmaxCrossEntropy()(x, t)

def softmax_cross_entropy_simple(x, t):
    x, t = as_variable(x), as_variable(t)
    N = x.shape[0] # データの数
    p = softmax_simple(x)
    p = clip(p, 1e-15, 1.0)  # log(0) を防ぐために，p の値を 1e-15 から 1.0 の間に収める
    log_p = log(p)
    tlog_p = log_p[np.arange(N), t.data] # t.data は教師ラベル
    y = -1 * sum(tlog_p) / N
    return y

def accuracy(y, t):
    y, t = as_variable(y), as_variable(t)
    pred = y.data.argmax(axis=1).reshape(t.shape)
    result = (pred == t.data)
    acc = (pred == t.data).mean()
    return Variable(as_array(acc))

class ReLU(Function):
    def forward(self, x):
        y = np.maximum(x, 0.0)
        return y
    
    def backward(self, gy):
        x, = self.inputs
        mask = x.data > 0
        gx = gy * mask
        return gx

def relu(x):
    return ReLU()(x)