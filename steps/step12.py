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
    def __call__(self, *inputs): # アスタリスクは可変長引数を表す
        '''
        入力が複数ある場合に対応させる
        '''
        xs = [x.data for x in inputs]
        ys = self.forward(*xs) # アスタリスクをつけてアンパッキングする
        if not (isinstance(ys, tuple)):
            ys = (ys,) # ys をタプルにする
        outputs = [Variable(as_array(y)) for y in ys]

        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs
        # リストの要素が1つの時は，最初の要素を返す
        return outputs if len(outputs) > 1 else outputs[0]
    
    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
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
    
class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return (y,) # タプルを返す

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

x0 = Variable(np.array(2))
x1 = Variable(np.array(3))
x2 = Variable(np.array(4))
f = Add()
# y = f(x0, x1, x2) # TypeError
y = f(x0, x1)
print(y.data)