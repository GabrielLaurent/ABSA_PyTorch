# src/data/vocabulary.py

class Vocabulary:
    def __init__(self):
        self.word2index = {}
        self.index2word = {}
        self.index = 0

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.index
            self.index2word[self.index] = word
            self.index += 1

    def __len__(self):
        return len(self.word2index)