import numpy as np

class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supportted'.format(type(data)))
        self.data = data
        self.grad = None
        self.creator = None # 変数にとって生みの親である関数
    
    def set_creator(self, func):
        '''
        creatorを指定しない変数も存在するので，set_creatorという関数が必要
        '''
        self.creator = func
    
    def backward(self):
        if self.grad is None: # grad の初期化に使用する
            self.grad = np.ones_like(self.data) # self.data と型も同じになるs

        funcs = [self.creator] # 次の関数を見つけるたびにここに追加される
        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad) # backward メソッドを呼び出す

            if x.creator is not None:
                funcs.append(x.creator) # ひとつ前の関数をリストに追加

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(as_array(y))
        output.set_creator(self) # 出力の変数に生みの親である関数を記憶させる
        self.input = input
        self.output = output # 出力も記憶する
        return output
    
    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()

# 手動でバックプロパゲーションを実装
class Square(Function):
    def forward(self, x):
        y = x **2
        return y
    
    def backward(self, gy):
        '''
        gy: 出力側から伝わる微分値
        '''
        x = self.input.data
        gx = 2 * x * gy
        return gx

class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y
    
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx
    
def square(x):
    f = Square()
    return f(x)

def exp(x):
    f = Exp()
    return f(x)

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

# ユニットテスト
import unittest

def numerical_diff(f, x, eps=1e-4):
    '''
    数値微分
    '''
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)

class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected)
    
    def test_backward(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        expected = np.array(6.0)
        self.assertEqual(x.grad, expected)

    def test_gradient_check(self):
        x = Variable(np.random.rand(1))
        y = square(x)
        y.backward()
        num_grad = numerical_diff(square, x)
        flg = np.allclose(x.grad, num_grad)
        self.assertTrue(flg)