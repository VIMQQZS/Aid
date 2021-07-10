from keras.preprocessing.text import Tokenizer

samples = ['我 在 哈尔滨 读书', '我 去 深圳 就职']

tokenizer = Tokenizer()
tokenizer.fit_on_texts(samples)

word_index = tokenizer.word_index
print(word_index)
print(len(word_index))

sequences = tokenizer.texts_to_sequences(samples)
print(sequences)

one_hot_results = tokenizer.texts_to_matrix(samples)
print(one_hot_results)