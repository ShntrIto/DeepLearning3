import numpy as np
import heapq as hq
import weakref
import contextlib
import dezero
from memory_profiler import profile

class Config:
    '''
    順伝播・逆伝播のモードを制御
    '''
    enable_backprop = True

class Variable:
    __array_priority__ = 200
    def __init__(self, data, name=None):
        '''
        name: 変数の名前，計算グラフの可視化に用いる
        '''
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supportted'.format(type(data)))
        self.data = data
        self.name = name # 代入した変数への名前ではなく，この数値が持つ名前
        self.grad = None
        self.creator = None # 変数にとって生みの親である関数
        self.generation = 0 # 正しい逆伝播のための世代
    
    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p + ')'

    def __mul__(self, other):
        '''
        * を使って，mul(x0, x1)を実行するために必要
        '''
        return mul(self, other)

    def __add__(self, other):
        return add(self, other)
    
    # @property は，ndarray の shape メソッド等を Variable のインスタンス変数として使う場合に必要
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype
    
    @property
    def T(self):
        return dezero.functions.transpose(self)

    def set_creator(self, func):
        '''
        creatorを指定しない変数も存在するので，set_creatorという関数が必要
        '''
        self.creator = func
        # input側から，output側へ set_creator 関数を実行するので，自分の関数の generation に 1 を加える
        self.generation = func.generation + 1
    
    def reshape(self, *shape):
        '''
        タプルやリスト，または引数をそのまま受け取った場合でも reshape が可能
        '''
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return dezero.functions.reshape(self, shape)

    def transpose(self, *axes):
        if len(axes) == 0:
            axes = None
        elif len(axes) == 1:
            if isinstance(axes[0], (tuple, list)) or axes[0] is None:
                axes = axes[0]
        return dezero.functions.transpose(self, axes)

    def backward(self, retain_grad=False, create_graph=False):
        '''
        retain_grad: 中間の変数に対して勾配を保持するかどうか
        '''
        if self.grad is None: # grad の初期化に使用する
            self.grad = Variable(np.ones_like(self.data)) # grad も Variable インスタンスとする

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

            with using_config('enable_backprop', create_graph):
                gxs = f.backward(*gys) # 本来の backward
                if not isinstance(gxs, tuple): # 単にgxに要素のみ（スカラ）が入っている場合でも，タプルに変換可能
                    gxs = (gxs,) # なぜタプルにするんだっけ？
                
                for x, gx in zip(f.inputs, gxs):
                    if x.grad is None:
                        x.grad = gx
                    else:
                        x.grad = x.grad + gx # それまでの勾配に，別の勾配を足し合わせる

                    if x.creator is not None:
                        add_func(x.creator)
            
            # なぜここで勾配を None にしても backward が上手くいくのか理解できていない
            # 誤差逆伝播を終えた output の勾配だけを None にしているかjら？
            if not retain_grad: 
                for y in f.outputs:
                    y().grad = None # yはweakrefなので値にアクセスするためにy()を使う

    def cleargrad(self):
        self.grad = None
    
    def sum(self, axis=None, keepdims=False):
        return dezero.functions.sum(self, axis, keepdims)

class Parameter(Variable):
    # Variable を継承するだけ
    # Variable と全く同じ機能を持つが，Parameter と Variable は区別ができる
    pass

class Function:
    def __call__(self, *inputs): # アスタリスクは可変長引数を表す
        '''
        関数として呼ばれた時に実行される関数s
        '''
        inputs = [as_variable(x) for x in inputs] # 必ず Variable インスタンスとなるようにする
        xs = [x.data for x in inputs]
        ys = self.forward(*xs) # アスタリスクをつけてアンパッキングする
        if not (isinstance(ys, tuple)):
            ys = (ys,) # ys をタプルにする
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs]) # 入力の中の，最も大きい generation に合わせる
            for output in outputs:
                output.set_creator(self) # 逆伝播用の計算グラフを作る（推論時には必要なし）
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
        gy: 出力k側から伝わる微分値
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
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 + x1
        return (y,) # タプルを返す
    
    def backward(self, gy):
        gx0, gx1 = gy, gy
        if self.x0_shape != self.x1_shape:
            gx0 = dezero.functions.sum_to(gx0, self.x0_shape)
            gx1 = dezero.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1 # 1つの入力に対して複数の出力

class Mul(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 * x1
        return y
    
    def backward(self, gy):
        x0, x1 = self.inputs # Variable インスタンスをそのまま使う
        gx0 = gy * x1
        gx1 = gy * x0
        if self.x0_shape != self.x1_shape:
            gx0 = dezero.functions.sum_to(gx0, self.x0_shape)
            gx1 = dezero.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1 # ここでも 「*演算子」が呼ばれ，計算グラフが構築される 

class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy

class Sub(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 - x1
        return y
    def backward(self, gy):
        gx0 = gy
        gx1 = -gy
        if self.x0_shape != self.x1_shape:
            gx0 = dezero.functions.sum_to(gx0, self.x0_shape)
            gx1 = dezero.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1

class Div(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 / x1
        return y
    
    def backward(self, gy):
        x0, x1 = self.inputs # Variable インスタンスをそのまま使う
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        if self.x0_shape != self.x1_shape:
            gx0 = dezero.functions.sum_to(gx0, self.x0_shape)
            gx1 = dezero.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1
class Pow(Function):
    def __init__(self, c):
        self.c = c # ここでは c を定数として扱う
    
    def forward(self, x):
        y = x ** self.c
        return y
    
    def backward(self, gy):
        x, = self.inputs # Variable インスタンスをそのまま使う
        c = self.c
        gx = c * x ** (c - 1) * gy
        return gx

# class Sin(Function):
#     def forward(self, x):
#         y = np.sin(x)
#         return y
#     def backward(self, gy):
#         x = self.inputs[0].data
#         gx = gy * np.cos(x)
#         return gx

@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)

def no_grad():
    return using_config('enable_backprop', False)

def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)

def add(x0, x1):
    x1 = as_array(x1)
    return Add()(x0, x1)

def mul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x0, x1)

# def sin(x):
#     return Sin()(x)

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

def neg(x):
    return Neg()(x)

def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)

def rsub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x1, x0)

def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)

def rdiv(x0, x1):
    x1 = as_array(x1)
    return Div()(x1, x0)

def as_variable(obj):
    '''
    Variableインスタンスに変換
    '''
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)

def pow(x, c):
    return Pow(c)(x)

def setup_variable():
    Variable.__add__  = add
    Variable.__radd__ = add
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow