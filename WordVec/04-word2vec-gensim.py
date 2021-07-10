import jieba
import re
from gensim.models import word2vec

# 读取停用词
stop_words = []
with open('chineseStopWords.txt','r',encoding='GBK') as f_reader:
    for line in f_reader:
        line = line.replace('\r','').replace('\n','').strip()
        stop_words.append(line)
print(len(stop_words))
stop_words = set(stop_words)
print(len(stop_words))

txtPath = 'data/天龙八部.txt'
txtPathNew = 'data/分词后的天龙八部.txt'
modelPath = 'data/天龙八部.model'

# 文本预处理
sentences = []
rules = u"[\u4e00-\u9fa5]+"
pattern = re.compile(rules)
f_writer = open('data/分词后的天龙八部.txt','w',encoding='utf-8')
with open('data/天龙八部.txt','r',encoding='utf-8') as f_reader:
    lines = f_reader.readlines()
    for line in lines:
        line = line.replace('\r','').replace('\n','').strip()
        if line == '' or line is None:
            continue
        line = ' '.join(jieba.cut(line))
        seg_list = pattern.findall(line)
        print(seg_list)
        word_list = []
        for word in seg_list:
            if word not in stop_words:
                word_list.append(word)
        if len(word_list) > 0:
            sentences.append(word_list)
            line = ' '.join(seg_list)
            f_writer.write(line+'\n')
            f_writer.flush()
f_writer.close()
print(sentences[:10])

# 模型训练
model = word2vec.Word2Vec(sentences, iter=100, size=300)
# 选出10个与乔峰最相近的10个词
for e in model.most_similar(positive=['乔峰'],topn=10):
    print(e[0],e[1])
# 加载语料
sentences2 = word2vec.Text8Corpus('data/分词后的天龙八部.txt')
print(sentences2)
# 训练模型
model = word2vec.Word2Vec(sentences2,iter=50,size=100)
# 选出10个与乔峰最相近的10个词
for e in model.most_similar(positive=['乔峰'],topn=10):
    print(e[0],e[1])

# 保存模型
model.save('data/天龙八部.model')
# 加载模型
model2 = word2vec.Word2Vec.load('data/天龙八部.model')
# 选出10个与乔峰最相近的10个词
for e in model.most_similar(positive=['乔峰'],topn=10):
    print(e[0],e[1])

#计算两个词语的相似度
sim_value = model.similarity('乔峰','萧峰')
print(sim_value)
#计算两个集合的相似度
list1 = ['乔峰','萧远山']
list2= ['慕容复','慕容博']
sim_value = model.n_similarity(list1,list2)
print(sim_value)
#选出集合中不同类型的词语
list3 = ['段誉','阿紫','王语嫣','丁春秋']
print(model.doesnt_match(list3))
#查看词向量值
print(type(model['乔峰']))
print(len(model['乔峰']))
print(model['乔峰'])