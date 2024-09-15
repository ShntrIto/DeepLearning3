import numpy as np
import heapq as hq
import weakref
import contextlib
from memory_profiler import profile

class Variable:
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

    def set_creator(self, func):
        '''
        creatorを指定しない変数も存在するので，set_creatorという関数が必要
        '''
        self.creator = func
        # input側から，output側へ set_creator 関数を実行するので，自分の関数の generation に 1 を加える
        self.generation = func.generation + 1

    def backward(self, retain_grad=False):
        '''
        retain_grad: 中間の変数に対して勾配を保持するかどうか
        '''
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
            
            # なぜここで勾配を None にしても backward が上手くいくのか理解できていない
            # 誤差逆伝播を終えた output の勾配だけを None にしているかjら？
            if not retain_grad: 
                for y in f.outputs:
                    y().grad = None # yはweakrefなので値にアクセスするためにy()を使う

    def cleargrad(self):
        self.grad = None
    

class Config:
    '''
    順伝播・逆伝播のモードを制御
    '''
    enable_backprop = True

@contextlib.contextmanager
def using_confikg(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)

def no_grad():
    return using_config('enable_backprop', False)

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

x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
print(x)
print('shape: ', x.shape)
print('len: ', len(x))