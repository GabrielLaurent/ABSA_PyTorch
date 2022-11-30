class Vocabulary:
    def __init__(self, tokens=None):
        self.token_to_index_map = {'<pad>': 0, '<unk>': 1}
        self.index_to_token_map = {0: '<pad>', 1: '<unk>'}
        self.next_index = len(self.token_to_index_map)

        if tokens is not None:
            self.build_vocabulary(tokens)

    def build_vocabulary(self, tokens):
        for token in tokens:
            if token not in self.token_to_index_map:
                self.token_to_index_map[token] = self.next_index
                self.index_to_token_map[self.next_index] = token
                self.next_index += 1

    def token_to_index(self, token):
        return self.token_to_index_map.get(token, 1)  # Return <unk> index if token is not in vocab

    def index_to_token(self, index):
        return self.index_to_token_map.get(index, '<unk>')

    def __len__(self):
        return len(self.token_to_index_map)