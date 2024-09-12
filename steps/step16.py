import numpy as np

class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supportted'.format(type(data)))
        self.data = data
        self.grad = None
        self.creator = None # 変数にとって生みの親である関数
        self.generation = 0 # 正しい逆伝播のための世代
    
    def set_creator(self, func):
        '''
        creatorを指定しない変数も存在するので，set_creatorという関数が必要
        '''
        self.creator = func
        # input側から，output側へ set_creator 関数を実行するので，自分の関数の generation に 1 を加える
        self.generation = func.generation + 1

    def backward(self):
        if self.grad is None: # grad の初期化に使用する
            self.grad = np.ones_like(self.data) # self.data と型も同じになる

        funcs = []
        seen_set = set() # 集合を使うことで，同じ関数を重複して追加することを避ける

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)
        
        add_func(self.creator)

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
                    add_func(x.creator)

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

        self.generation = max([x.generation for x in inputs]) # 入力の中の，最も大きい generation に合わせる
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

x = Variable(np.array(2.0)) # x
a = square(x) # x^2
y = add(square(a), square(a)) # (x^2)^2 + (x^2)^2 = 2x^4
y.backward()

print(y.data)
print(x.grad) # 8x^3