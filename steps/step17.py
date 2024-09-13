import numpy as np
import heapq as hq
import weakref
from memory_profiler import profile

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

        def add_func_heap(f):
            if f not in seen_set:
                hq.heappush(funcs, (-1*f.generation, f))
                seen_set.add(f)
        
        add_func(self.creator)
        # add_func_heap(self.creator)

        while funcs:
            f = funcs.pop()
            # f = hq.heappop(funcs)[1]
            # print(type(f))

            gys = [output().grad for output in f.outputs] # 弱参照を呼び出す場合には output() とする
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
        self.outputs = [weakref.ref(output) for output in outputs] # 循環参照の解消のためにoutputへの弱参照を追加
        # リストの要素が1つの時は，最初の要素を返す
        return outputs if len(outputs) > 1 else outputs[0]
    
    def __lt__(self, other): # Function()のインスタンス同士の大小確認
        return self.generation < other.generation

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

@profile
def memory_check():
    for i in range(10):
        x = Variable(np.random.randn(10000))
        y = square(square(square(x)))
        # 次の計算を行うとき，メモリが解放され，xとyが上書きされる

memory_check()