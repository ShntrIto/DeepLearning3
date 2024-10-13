import weakref
import numpy as np
import dezero.functions as F
from dezero.core import Parameter
from dezero import cuda
from dezero.utils import pair

class Layer:
    def __init__(self):
        self._params = set() # Layer インスタンスが持つパラメータを保持する
    
    def __setattr__(self, name, value):
        # インスタンス変数を設定する際に呼び出される特殊メソッド
        if isinstance(value, (Parameter, Layer)):
            self._params.add(name) # インスタンス名を保存
        super().__setattr__(name, value)

    def __call__(self, *inputs):
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(y) for y in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, inputs):
        raise NotImplementedError()

    def params(self):
        for name in self._params:
            obj = self.__dict__[name]

            if isinstance(obj, Layer):
                # obj が Layer インスタンスの場合は，params() を呼び出すことで入れ子構造にする
                yield from obj.params()
            else:
                yield obj

    def cleargrads(self):
        for param in self.params():
            param.cleargrad()
    
    def _flatten_params(self, params_dict, parent_key=""):
        for name in self._params:
            obj = self.__dict__[name]
            # REVIEW: これだと parent_key の有無にかかわらず / が付くのではない？
            key = parent_key + '/' + name if parent_key else name 

            if isinstance(obj, Layer):
                obj._flatten_params(params_dict, key)
            else:
                params_dict[key] = obj
    
    def save_weights(self, path):
        self.to_cpu()
        
        params_dict = {}
        self._flatten_params(params_dict)
        array_dict = {key: param.data for key, param in params_dict.items() 
                      if param is not None}
        # ファイルの作成時には，user interrupt が発生することを想定して
        # try 文を使う
        try:
            np.savez_compressed(path, **array_dict)
        except(Exception, KeyboardInterrupt) as e:
            if os.path.exists(path):
                os.remove(path)
            raise
    
    def load_weights(self, path):
        npz = np.load(path)
        params_dict = {}
        self._flatten_params(params_dict)
        for key, param in params_dict.items():
            param.data = npz[key]
    
    def to_cpu(self):
        # Linear クラスのパラメータを CPU に移動させる
        for param in self.params():
            param.to_cpu()
    
    def to_gpu(self):
        # Linear クラスのパラメータを GPU に移動させる
        for param in self.params():
            param.to_gpu()

class Linear(Layer):
    def __init__(self, out_size, nobias=False, dtype=np.float32, in_size=None):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype
        
        self.W = Parameter(None, name='W')
        if self.in_size is not None: # in_size の指定がない場合は後回し（指定することもできる！）
            self._init_W()
        
        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_size, dtype=dtype), name='b')
    
    def _init_W(self, xp=np):
        I, O = self.in_size, self.out_size
        W_data = xp.random.randn(I, O).astype(self.dtype) * np.sqrt(1 / I) # 変数をわざわざ分けているのは可読性のため？
        self.W.data = W_data

    def forward(self, x):
        # データを流すタイミング（linearの実行時）で重みを初期化
        if self.W.data is None:
            self.in_size = x.shape[1]
            xp = cuda.get_array_module(x)
            self._init_W(xp)

        y = F.linear(x, self.W, self.b)
        return y
    

class Conv2d(Layer):
    def __init__(self, out_channels, kernel_size, stride=1, 
                 pad=0, nobias=False, dtype=np.float32, in_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.dtype = dtype

        self.W = Parameter(None, name='W')
        if self.in_channels is not None:
            self._init_W()
        
        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_channels, dtype=dtype), name='b')
    
    def _init_W(self, xp=np):
        C, OC = self.in_channels, self.out_channels
        KH, KW = pair(self.kernel_size)
        scale = np.sqrt(1 / (C * KH * KW))
        W_data = xp.random.randn(OC, C, KH, KW).astype(self.dtype) * scale
        self.W.data = W_data
    
    def forward(self, x):
        if self.W.data is None:
            self.in_channels = x.shape[1] # (B, C, H, W) から C を取り出す
            xp = cuda.get_array_module(x)
            self._init_W(xp)
        
        y = F.conv2d_simple(x, self.W, self.b, stride=self.stride, pad=self.pad)
        # y = F.conv2d(x, self.W, self.b, stride=self.stride, pad=self.pad)
        return y

class RNN(Layer):
    def __init__(self, hidden_size, in_size=None):
        super().__init__()
        self.x2h = Linear(hidden_size, in_size=in_size)
        self.h2h = Linear(hidden_size, in_size=hidden_size, nobias=True) # bias は一つでいい
        self.h = None
    
    def reset_state(self):
        self.h = None
        
    def forward(self, x):
        if self.h is None:
            h_new = F.tanh(self.x2h(x))
        else:
            h_new = F.tanh(self.x2h(x) + self.h2h(self.h)) # ひとつ前の h に重みを掛けて足し合わせる
        return h_new