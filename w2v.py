import numpy as np
import gensim
import os


class MySentence:
    def __init__(self, files):
        self.files = files

    def __iter__(self):
        for file in self.files:
            with open(file, 'r', encoding='utf8') as fp:
                for line in fp.readlines():
                    # 预处理代码写在这里
                    yield line.strip().split(' ')


def train_embedding(files, save_path):
    sentences = MySentence(files)
    model = gensim.models.Word2Vec(sentences)
    model.save(save_path)
    return model


def load_model(model_path):
    if not os.path.exists(model_path):
        return None
    return gensim.models.Word2Vec.load(model_path)


def load_embedding_matrix(model, vocab, embedding_size):
    if not model:
        print('no w2v model, normal init word embedding')
        return np.random.normal(size=[len(vocab), embedding_size])

    vocab_size = len(vocab)
    embeddings = [[0 for i in range(embedding_size)] for j in range(vocab_size)]
    for word, word_id in vocab.items():
        try:
            word_embedding = model[word]
        except KeyError:
            word_embedding = np.random.normal(loc=0, scale=1, size=[embedding_size, ])
        embeddings[word_id] = word_embedding
    print('init word embedding using w2v model')
    return np.asarray(embeddings)


model = train_embedding(files=['./DataManager/datasets/restaurant/train_clean.txt'], save_path='./models/w2v.model')

print(model['apple'])
print(model.most_similar('apple'))