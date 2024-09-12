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
            self.grad = np.ones_like(self.data) # self.data と型も同じになる

        funcs = [self.creator] # 次の関数を見つけるたびにここに追加される
        while funcs:
            f = funcs.pop()

            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)

            if not isinstance(gxs, tuple): # 単にgxに要素のみ（スカラ）が入っている場合でも，タプルに変換可能
                gxs = (gxs,) # なぜタプルにするんだっけ？
            
            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx # それまでの勾配に，別の勾配を足し合わせる

                if x.creator is not None:
                    funcs.append(x.creator)

    def cleargrad(self):
        self.grad = None

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
        x = self.inputs[0].data # 入力がnp.arrayだから？なぜ0番目？
        gx = 2 * x * gy
        return gx

class Exp(Function):
    def forward(self, x):
        y = inp.exp(x)
        return y
    
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx
    
class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return (y,) # タプルを返す
    
    def backward(self, gy):
        return gy, gy # 1つの入力に対して複数の出力

def square(x):
    f = Square()
    return f(x)

def exp(x):
    f = Exp()
    return f(x)

def add(x0, x1):
    f = Add()
    return f(x0, x1)

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

a = Variable(np.array(3.0))
b = add(a, a)
print(b.data)
b.backward()
print('a.grad:', a.grad)

a.cleargrad()
b = add(add(a, a), a)
b.backward()
print('a.grad:', a.grad)