import torch
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self, vocab, config):
        super(BaseModel, self).__init__()
        self.embedding_dim = config['embedding_dim']
        self.hidden_dim = config['hidden_dim']
        self.num_layers = config['num_layers']
        self.dropout = config['dropout']
        self.vocab = vocab

        self.embedding = nn.Embedding(len(vocab), self.embedding_dim, padding_idx=0)

        # Load pre-trained embeddings if available
        if vocab.embedding_matrix is not None:
            self.embedding.weight.data.copy_(vocab.embedding_matrix)
            if not config['fine_tune_embeddings']:
                self.embedding.weight.requires_grad = False

        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, num_layers=self.num_layers, dropout=self.dropout, batch_first=True)
        self.dropout_layer = nn.Dropout(self.dropout)

        self.fc = nn.Linear(self.hidden_dim, len(config['aspect_categories']))

    def forward(self, text):
        embedded = self.embedding(text)
        output, (hidden, cell) = self.lstm(embedded)
        output = self.dropout_layer(output[:, -1, :])  # Use the last hidden state
        output = self.fc(output)
        return output
