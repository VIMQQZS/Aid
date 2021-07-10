# word2vec 参考tensorflow官网提供的example代码编写我word2vec
#   https://github.com/tensorflow/tensorflow/blob/r0.12/tensorflow/examples/tutorials/word2vec/word2vec_basic.py
import os
import math
import zipfile
import collections
import tensorflow as tf
import numpy as np
import random
import pickle

from six.moves import urllib

url = 'http://mattmahoney.net/dc/' # 数据下载链接


# step 1.下载数据
def get_data(filename = 'text8.zip', expected_bytes = 31344016):
    '''
    获得数据从文件中，如果文件不存在则进行下载
    :return:
    '''
    if not os.path.exists(filename):
        filename, _ = urllib.request.urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception('failed to verify', filename, '.')

    return filename

# 获得文件名
filename = get_data()


def read_data(filename):
    '''
    提取数据从zip文件的第一个文件中，返回一个list单词列表
    :param filename:
    :return:
    '''
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split() # compat.as_str将二进制str转换为文本str
    return data


words = read_data(filename)
print('Data size', len(words)) # Data size 17005207


# step 2. 建立字典并且用UNK替代稀有单词
vocabulary_size = 50000


def build_dataset(words):
    '''
    建立数据集
    :param words:
    :return:
    '''
    count = [['UNK', -1]]
    # 得到最常用的vocabulary_size-1个单词，(词，词频)
    count.extend(collections.Counter(words).most_common(vocabulary_size-1))
    # 建立词典
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    # 获取每一个单词的索引值
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    # 更新UNK的计数
    count[0][1] = unk_count
    # 颠倒字典的元素
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

    # 返回单词的索引值data list和words对应，每个单词的计数count dict，字典(word, index)，逆字典(index,word)
    return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(words)
del words
with open('reverse_dictionary.dict.pkl', 'wb') as output:
    pickle.dump(reverse_dictionary, output)
print('Most common words (+UNK)', count[:5])
print('First 10 data', data[:10], [reverse_dictionary[i] for i in data[:10]])


# step 3.生成训练batch对于skip-gram模型
data_index = 0
def generate_batch(batch_size, num_skips, skip_window):
    '''
    生成训练数据
    :param batch_size: 表示每个批次大小
    :param num_skips: skip的数量，就是从上下文窗口采样的数量，batch_size%num_skips == 0为true
    :param skip_window: 窗口大小，单方向的，2*skip_window需要大于等于num_skips
    :return:
    '''
    global data_index

    assert batch_size % num_skips == 0
    assert num_skips <= 2* skip_window

    batch = np.ndarray(shape=[batch_size], dtype=np.int32)
    labels = np.ndarray(shape=[batch_size, 1], dtype=np.int32)

    span = 2*skip_window+1
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index+1) % len(data)

    for i in range(batch_size // num_skips):
        target = skip_window # 输入位置，中间位置
        target_to_avoid = [skip_window] # 不能选择的位置
        for j in range(num_skips):
            while target in target_to_avoid:
                target = random.randint(0, span-1) # 选择一个不在避免位置的
            target_to_avoid.append(target) # 添加到不可选择
            # 添加一个输入和标签
            batch[i*num_skips+j] = buffer[skip_window]
            labels[i*num_skips+j, 0] = buffer[target]
        # 加入一个新的词，索引+1
        buffer.append(data[data_index])
        data_index = (data_index+1) % len(data)

    return batch, labels

batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
    print(batch[i], reverse_dictionary[batch[i]], '->', labels[i,0], reverse_dictionary[labels[i, 0]])

# step 4.建立和训练skip-gram模型

batch_size = 128
embedding_size = 128 # 词向量的维度
skip_window = 1      # 左右窗口的大小，考虑词的个数
num_skips = 2        # 采样数量，每一个输入对应的label的数量

# 随机选择验证集采样最近的邻居，从前100个里随机选择16个
valid_size = 16
valid_window = 100
## 是从a 中以概率P，随机选择3个, p没有指定的时候相当于是一致的分布
valid_examples = np.random.choice(a=valid_window, size=valid_size, replace=False, p=None)
num_sampled = 64 # 负采样数量

graph = tf.Graph()

with graph.as_default():
    # 输入数据
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1]) # 为什么多出来一维，因为下面nce函数的规定输入维度
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)# [16,]

    with tf.device('/cpu:0'):
        # LookUp嵌入表
        embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        # 获得词嵌入向量
        ## tf.nn.embedding_lookup函数的用法主要是选取一个张量里面索引对应的元素。train_input是训练的输入索引，得到对应索引的向量
        embed = tf.nn.embedding_lookup(embeddings, train_inputs) # 维度[batch_size, D]

        nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0/math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

        # 直接集成了输出层的输出和损失的计算，同时考虑了负样本的计算
        loss = tf.reduce_mean(tf.nn.nce_loss(
            weights = nce_weights,
            biases = nce_biases,
            inputs = embed,
            labels = train_labels,
            num_sampled = num_sampled,
            num_classes = vocabulary_size
        ))

        # 构造优化器，SGD，lr=1.0
        train = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

        # 计算余弦相似度在minibatch和embedding之间
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True)) # 按行的维度求和，保持维度
        # # 'x' is [[1, 1, 1]
        # #         [1, 1, 1]]
        # # 求和
        # tf.reduce_sum(x) == > 6
        # # 按列求和
        # tf.reduce_sum(x, 0) == > [2, 2, 2]
        # # 按行求和
        # tf.reduce_sum(x, 1) == > [3, 3]
        normalized_embeddings = embeddings/norm # 归一化处理,[V,D]
        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset) # [16, D]
        # 相似度，16个验证向量和所有的embedding的余弦相似度
        similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True) # 第二个参数转置

        init = tf.global_variables_initializer()

# step 5.开始训练，训练很快
## average loss at step 100000 : 4.674877667903901
num_steps = 100001

average_loss = 0
with tf.Session(graph=graph) as session:
    init.run()
    print('Initialized')

    for i in range(num_steps):
        # print(num_steps, i)
        batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
        feed_dict = {train_inputs:batch_inputs, train_labels:batch_labels}

        _, loss_val = session.run([train, loss], feed_dict=feed_dict)
        average_loss += loss_val

        # 显示损失
        if i % (num_steps//50) == 0 and i >  0:
            average_loss /= (num_steps//50)
            print('average loss at step', i, ':', average_loss)
            average_loss = 0

        # 显示相似度
        if i % (num_steps//2) == 0 and i > 0:
            sim = similarity.eval() # eval和run的作用基本一样
            for j in range(valid_size):
                valid_word = reverse_dictionary[valid_examples[j]]
                top_k = 8
                # print('=====================',sim.shape)
                nearest = (-sim[j, :]).argsort()[1:top_k+1] # 符号的目的是为了从大到小建立索引（默认从小到大）
                # 不考虑第0个，本身
                # >> > x = np.array([3, 1, 2])
                # >> > np.argsort(x)
                # array([1, 2, 0])
                print('step {} nearest to "{}":'.format(i, valid_word))
                for k in range(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    print(k+1, close_word, end=';')
                print()
    final_embeddings = normalized_embeddings.eval()
    np.save('embeddings.npy', final_embeddings)
    print('train finished.')