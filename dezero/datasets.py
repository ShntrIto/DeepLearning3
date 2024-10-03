import numpy as np

class Dataset:
    def __init__(self, train=True, transform=None, target_transform=None):
        # transform や target_transform は呼び出し可能なオブジェクト（関数）である必要がある
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        # 前処理がない場合はそのままデータを返すだけ
        if self.transform is None:
            self.transform = lambda x: x # 恒等変換?
        if self.target_transform is None:
            self.target_transform = lambda x: x # transform と何違うのか（教師ラベルに対する処理）

        self.data = None
        self.label = None # label? データに対応するラベルを割り当てる
        self.prepare()
    
    def __getitem__(self, index): # getitem? 
        # [] でアクセスしたときの挙動を設定できる
        assert np.isscalar(index)
        if self.label is None: # 教師無しの場合（ラベルが None の場合）でも対応できる
            return self.transform(self.data[index]), None
        else:
            return sel.transform(self.data[index]), self.target_transform(self.label[index])
        
    def __len__(self):
        # データセットの"長さ"って，多次元配列の場合どの"長さ"なの？
        # → 最初の軸に沿った要素の数が出力される
        return len(self.data)
    
    def prepare(self):
        # 継承先のクラスでデータの準備を実装するための関数
        pass # pass?

def get_spriral(train=True):
    # dezero からそのまま引用
    seed = 1984 if train else 2020
    np.random.seed(seed)

    num_data, num_class, input_dim = 100, 3, 2
    data_size = num_data * num_class
    x = np.zeros((data_size, input_dim), dtype=np.float32)
    t = np.zeros(data_size, dtype=np.int)
    
    for j in range(num_class):
        for i in range(num_data):
            rate = i / num_data
            radius = 1.0 * rate
            theta = j * 4.0 + 4.0 * rate + np.random.randn() * 0.2
            ix = num_data * j + i
            x[ix] = np.array([radius * np.sin(theta),
                              radius * np.cos(theta)]).astype(np.float32)
            t[ix] = j
    
    indices = np.random.permutation(num_data * num_class) # permutation?
    x = x[indices]
    t = t[indices]
    return x, t

class Spiral(Dataset):
    def prepare(self):
        self.data, self.label = get_spriral(self.train)

