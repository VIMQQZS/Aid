from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import re
import math
import random
import jieba
# jieba 读取停用词使用的库
import numpy as np
from six.moves import xrange
import tensorflow as tf

# 读取停用词
stop_words = []
with open('chineseStopWords.txt','r',encoding='GBK') as f_stopword:
    for line in f_stopword:
        line = line.replace('r', '').replace('\n','').strip()
        stop_words.append(line)
stop_words = set(stop_words)
print(len(stop_words))

# 文本预处理
raw_word_list = []
rules = u"([\u4e00-\u9fa5]+)"
# 汉字正则表达式
pattern = re.compile(rules)
# 生成正则化规则
f_writer = open("data/分词后的笑傲江湖.txt",'w',encoding='utf-8')
# count = 0
with open('data/笑傲江湖.txt','r',encoding='utf-8') as f_reader:
    lines = f_reader.readlines()
    for line in lines:
        line = line.replace('\r','').replace('\n','').strip()
        if line == '' or line is None:
            continue
        line = ' '.join(jieba.cut(line))
        seg_list = pattern.findall(line)
        print(seg_list)
        # count = count + 1
        # if count == 10:
        #     break
        word_list = []
        for word in seg_list:
            if word not in stop_words:
                word_list.append(word)
        if len(word_list) > 0:
            raw_word_list.extend(word_list)
            # extend()  函数可合并 1维列表为 1维列表
            line = ' '.join(seg_list)
            # str.join(sequence)   sequence -- 要连接的元素序列   返回通过指定字符连接序列中元素后生成的新字符串
            f_writer.write(line+'\n')
            f_writer.flush()
f_writer.close()
print(len(raw_word_list))
vocabulary_size = len(set(raw_word_list))
print(len(set(raw_word_list)))

words = raw_word_list
count = [['UNK','-1']]
count.extend(collections.Counter(words).most_common(vocabulary_size-1))
# extend 列表添加到列表    Counter 计数器   most_common 最多到第几位
print('count',len(count))
dictionary = dict()
for word,_ in count:
    dictionary[word] = len(dictionary)
    # 通过 dictionary的长度编码

data = list()
unk_count = 0
for word in words:
    if word in dictionary:
        index = dictionary[word]
    else:
        index = 0
        unk_count = unk_count + 1
    data.append(index)
count[0][1] = unk_count
reverse_dictionary = dict(zip(dictionary.values(),dictionary.keys()))
# zip() 将 dictionary 中的 value值和 key值打包成元组
del words

# print(reverse_dictionary[1000])
print(data[:200])

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
    # assert 断言

    batch = np.ndarray(shape=[batch_size], dtype=np.int32)
    labels = np.ndarray(shape=[batch_size, 1], dtype=np.int32)

    span = 2*skip_window+1     ## 3
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index+1) % len(data)
    ## 到这一步的时候data_index=3, buffer中是[5234, 3081, 12]

    for i in range(batch_size // num_skips):
        target = skip_window     ## input word, 是buffer的中间位置
        target_to_avoid = [skip_window]    ## 记录已经选择的位置列表
        for j in range(num_skips):
            while target in target_to_avoid:
                target = random.randint(0, span-1)     ## 选择一个不在target_to_avoid的单词
            target_to_avoid.append(target)    ## 添加到已选择列表
            # 添加一个输入和标签
            batch[i*num_skips+j] = buffer[skip_window]    ## input word
            # buffer[skip_window]  skip_window是角标, 从buffer中依次弹出
            labels[i*num_skips+j, 0] = buffer[target]     ## output word
            # buffer[target]  target是角标, 从buffer中依次弹出
        ## 加入一个新的词，索引+1
        buffer.append(data[data_index])      ## 此时buffer中是[3081, 12, 6]
        data_index = (data_index+1) % len(data)   ## data_index = 4

    return batch, labels

'''
num_skips与skip_window之间的关系
很多人都不理解num_skips与skip_window之间的关系，skip_window这个参数限制了采样的范围，
skip_window=1就是在输入单词的左右各一个单词范围内采样，skip_window=2就是在输入单词的左右各2个单词的范围内采样，
num_skips参数是在skip_window规定的范围内采样多少个，比如skip_window=2的时候总共可以采样4个（input, output）单词对，
num_skips=2就表示在4个单词对中选择2个单词对作为训练数据。

generate_batch函数理解
变量说明：
假设batch_size=8，num_skips=2，skip_window=1。
indexs：中存的是要训练的单词的id。
buffer：是一个长度为2×skip_window+1的滑动窗口。

假设 indexs = [ 5234 , 3081 , 12 , 6 , 195 , 2 , 3134 , ... ] 

所以看到这里可以看出，index相当于是一个纸带，buffer是一个滑动窗口在上面移动，每次移动一个单词的长度，
而data_index就相当于是一个指针，指定buffer移动的下一个单词。
'''

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

# 校验集
# valid_word = ['令狐冲','左冷禅','林平之','岳不群','桃根仙']
# valid_examples = [dictionary[li] for li in valid_word]
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

        # 构造优化器，SGD，learn_rate=1.0 (学习率)
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


# Step 6: Visualize the embeddings.
def plot_with_labels(low_dim_embs, labels, filename='tsne.png', fonts=None):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     fontproperties=fonts,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.savefig(filename, dpi=600)


try:
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties

    # 为了在图片上能显示出中文
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    plot_only = 500
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
    labels = [reverse_dictionary[i] for i in xrange(plot_only)]
    plot_with_labels(low_dim_embs, labels, fonts=font)

except ImportError:
    print("Please install sklearn, matplotlib, and scipy to visualize embeddings.")