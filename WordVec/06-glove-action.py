from mxnet import nd
from mxnet.contrib import text

glove_vec = text.embedding.get_pretrained_file_names('glove')
print(glove_vec)
fasttext_vec = text.embedding.get_pretrained_file_names('fasttext')
print(fasttext_vec)

glove_6b50d = text.embedding.create('glove',pretrained_file_name='glove.6B.50d.txt')

word_size = len(glove_6b50d)
print(word_size)

# 词的索引
index = glove_6b50d.token_to_idx['happy']
print(index)
# 索引到词
word = glove_6b50d.idx_to_token[1752]
print(word)
# 词向量
print(glove_6b50d.idx_to_vec[1752])


# 余弦相似度
def cos_sim(x,y):
    return nd.dot(x,y)/(x.norm()*y.norm())
# nd.dot 按元素乘法(内积) nd.dot是矩阵乘法
# n = norm(v) 返回向量 v 的欧几里德范数。此范数也称为 2-范数、向量模或欧几里德长度。
# K>> norm([3 4])
# ans = 5

a = nd.array([4,5])
b = nd.array([400,500])
print(cos_sim(a,b))

# 求近义词
def norm_vecs_by_row(x):
    # 分母添加1e-10是为了数值稳定性
    return x/(nd.sum( x**2 , axis=1) + 1e-10).sqrt().reshape((-1, 1))

def get_knn(token_embedding, k, word):
    word_vec = token_embedding.get_vecs_by_tokens([word]).reshape((-1, 1))
    vocab_vecs = norm_vecs_by_row(token_embedding.idx_to_vec)
    dot_prod = nd.dot(vocab_vecs,word_vec)
    indices = nd.topk(dot_prod.reshape((len(token_embedding),)), k=k+1, ret_typ='indices')
    indices = [int(i.asscalar()) for i in indices]
    # 除去输入词
    return token_embedding.to_tokens(indices[1:])

sim_list = get_knn(glove_6b50d, 10, 'baby')
print(sim_list)

sim_val = cos_sim(glove_6b50d.get_vecs_by_tokens('baby'),glove_6b50d.get_vecs_by_tokens('babies'))
print(sim_val)

print(get_knn(glove_6b50d, 10, 'computer'))
print(get_knn(glove_6b50d, 10, 'run'))
print(get_knn(glove_6b50d, 10, 'love'))

# 求类比词
# vec(c)+vec(b)-vec(a)
def get_top_k_by_analogy(token_embedding, k, word1, word2, word3):
    word_vecs = token_embedding.get_vecs_by_tokens([word1, word2, word3])
    word_diff = (word_vecs[1] - word_vecs[0] + word_vecs[2]).reshape((-1, 1))
    vocab_vecs = norm_vecs_by_row(token_embedding.idx_to_vec)
    dot_prod = nd.dot(vocab_vecs, word_diff)
    indices = nd.topk(dot_prod.reshape((len(token_embedding),)), k=k, ret_typ='indices')
    indices = [int(i.asscalar()) for i in indices]
    return token_embedding.to_tokens(indices)

def cos_sim_word_analogy(token_embedding, word1, word2, word3, word4):
    words = [word1, word2, word3, word4]
    vecs = token_embedding.get_vecs_by_tokens(words)
    return cos_sim(vecs[1] - vecs[0] + vecs[2], vecs[3])

word_list = get_top_k_by_analogy(glove_6b50d, 1, 'man', 'woman', 'son')
print(word_list)

word_list = get_top_k_by_analogy(glove_6b50d, 1, 'man', 'son', 'woman')
print(word_list)

sim_val = cos_sim_word_analogy(glove_6b50d, 'man', 'woman', 'son', 'daughter')
print(sim_val)

word_list = get_top_k_by_analogy(glove_6b50d, 1, 'beijing', 'china', 'tokyo')
print(word_list)

word_list = get_top_k_by_analogy(glove_6b50d, 1, 'bad', 'worst', 'big')
print(word_list)

