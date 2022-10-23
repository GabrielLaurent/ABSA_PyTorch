import torch
import numpy as np

class Vocabulary:
    def __init__(self, data, config):
        self.word2idx = {'<pad>': 0, '<unk>': 1}
        self.idx2word = {0: '<pad>', 1: '<unk>'}
        self.word_counts = {}
        self.config = config
        self.embedding_matrix = None
        self.build_vocabulary(data)


    def build_vocabulary(self, data):
        for sentence in data:
            for word in sentence.split():
                if word not in self.word_counts:
                    self.word_counts[word] = 0
                self.word_counts[word] += 1

        for word, count in self.word_counts.items():
            if word not in self.word2idx:
                index = len(self.word2idx)
                self.word2idx[word] = index
                self.idx2word[index] = word


    def load_pretrained_embeddings(self):
        print("Loading pretrained embeddings from {}".format(self.config['pretrained_embeddings_path']))
        embeddings_index = {}
        with open(self.config['pretrained_embeddings_path'], encoding='utf8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs

        print('Found %s word vectors.' % len(embeddings_index))

        embedding_matrix = np.zeros((len(self.word2idx), self.config['embedding_dim']))
        for word, i in self.word2idx.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector

        self.embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float)
        print("Pretrained embeddings loaded")


    def __len__(self):
        return len(self.word2idx)
