import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None # 変数にとって生みの親である関数
    
    def set_creator(self, func):
        '''
        creatorを指定しない変数も存在するので，set_creatorという関数が必要
        '''
        self.creator = func
    
    def backward(self):
        f = self.creator # 関数の取得
        if f is not None:
            x = f.input # 関数の入力を取得
            x.grad = f.backward(self.grad) # 関数のbackwardメソッドを使って勾配を計算
            x.backward() # 自分よりひとつ前の変数のbackwardメソッドを再帰的に呼ぶ（creatorがNoneの時に終了）

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
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

A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b) # y = (exp(x^2))^2

assert y.creator == C
assert y.creator.input == b
assert y.creator.input.creator == B
assert y.creator.input.creator.input == a
assert y.creator.input.creator.input.creator == A
assert y.creator.input.creator.input.creator.input == x

y.grad = np.array(1.0)

C = y.creator # 関数の取得
b = C.input # 関数の入力を取得
b.grad = C.backward(y.grad) # 逆伝播する

B = b.creator
a = B.input
a.grad = B.backward(b.grad)

A = a.creator
x = A.input
x.grad = A.backward(a.grad)

## 自動バックプロパゲーションを試す
A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b) # y = (exp(x^2))^2

print('x.grad (forward): ', x.grad)
y.grad = np.array(1.0)
y.backward()
print('x.grad (backwarded): ',x.grad)
