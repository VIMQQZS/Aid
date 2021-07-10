import jieba
import fasttext

# 读取停用词
stop_words = []
with open('chineseStopWords.txt','r',encoding='GBK') as f_reader:
    for line in f_reader:
        line = line.replace('\n', '').replace('\r','').strip()
        stop_words.append(line)
print(len(stop_words))
stop_words = set(stop_words)
print(len(stop_words))


# 文本预处理
category = []
f_writer = open('data/new.test.txt','w',encoding='utf-8')

with open('data/test.txt','r',encoding='utf-8') as f_reader:
    for line in f_reader:
        line = line.replace('\n', '').replace('\r','').strip()
        line_list = line.split('\t')
        if len(line_list) == 2:
            seg_list = jieba.cut(line_list[1])
            word_list = []
            for word in seg_list:
                if word not in stop_words:
                    word_list.append(word)
            line = ' '.join(word_list)
            line = '__label__' + line_list[0] + '\t' + line + '\n'
            f_writer.write(line)
            f_writer.flush()
            if line_list[0] not in category:
                category.append(line_list[0])
        print(len(line_list))

print(category)

# 文本分类训练
model = fasttext.supervised('data/train.txt','classifier.model')
train_result = model.test('data/train.txt')
# 训练集train的准确率和召回率
print(':',train_result.precision)
print(':',train_result.recall)
test_result = model.test('data/test.txt')
# 测试集test的准确率和召回率
print('test:',test_result.precision)
print('test:',test_result.recall)

# 预测文本标签
texts = []
labels = model.predict(texts)
print(labels)
labels_proba = model.proba(texts)
print(labels_proba)
# 预测文本标签 k 表示 前k项
labels = model.predict(texts,k=3)
print(labels)
labels_proba = model.proba(texts,k=3)
print(labels_proba)


word2vec = fasttext.skipgram('data/分词后的天龙八部.txt','model')
# 维度
print(word2vec.dim)
print(word2vec.cosine_similarity(first_word='萧峰',second_word='乔峰'))
print(word2vec.cosine_similarity(first_word='萧峰',second_word='萧远山'))
print(word2vec.cosine_similarity(first_word='段誉',second_word='乔峰'))
# cosine_similarity()比较两个词之间的相似度
word2vec = fasttext.cbow('data/分词后的天龙八部.txt','model',dim=50)
# 维度
print(word2vec.dim)
print(word2vec.cosine_similarity(first_word='萧峰',second_word='乔峰'))
print(word2vec.cosine_similarity(first_word='萧峰',second_word='萧远山'))
print(word2vec.cosine_similarity(first_word='段誉',second_word='乔峰'))
# cosine_similarity()比较两个词之间的相似度
print(word2vec.words)