import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data

data = np.array(1.0)
x = Variable(data)
print(x.data)

x.data = np.array(4.0)
print(x.data)