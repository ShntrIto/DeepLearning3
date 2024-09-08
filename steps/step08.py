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

y.grad = np.array(1.0)
y.backward()
print(x.grad)