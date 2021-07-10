import numpy as np

samples = ['我 在 哈尔滨 读书', '我 去 深圳 就职']
toke_index = {}
for sample in samples:
    for word in sample.split():
        if word not in toke_index:
            toke_index[word] = len(toke_index) + 1

print(len(toke_index))
print(toke_index)

result = np.zeros(shape=(len(samples),len(toke_index)+1))
for i, sample in enumerate(samples):
    for _, word in list(enumerate(sample.split())):
        index = toke_index.get(word)
        # 字典中的 get()函数
        result[i,index] = 1

print(result)