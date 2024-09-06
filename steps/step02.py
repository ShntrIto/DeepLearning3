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


x = Variable(np.array(10))
f = Function()
y = f(x)

print(type(y))
print(y.data)

x = Variable(np.array(12))
f = Square()
y = f(x)
print(type(y))
print(y.data)