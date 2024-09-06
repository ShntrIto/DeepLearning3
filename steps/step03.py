import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data

class Function: # 基底クラスとしての Function
    def __call__(self, input):
        x = input.data # Variable インスタンスであることを仮定している
        y = self.forward(x)
        output = Variable(y)
        return output
    def forward(self, x):
        raise NotImplementedError()

class Square(Function):
    def forward(self, x):
        return x ** 2

class Exp(Function):
    '''
    exponential を計算する
    '''
    def forward(self, x):
        return np.exp(x)

A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x) # x^2 
b = B(a) # exp(x^2)
y = C(b) # (exp(x^2))^2
print(y.data)
print(type(y))