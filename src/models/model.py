import torch
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, embedding_weights=None):
        super(BaseModel, self).__init__()
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if embedding_weights is not None:
            self.embedding.weight.data.copy_(embedding_weights)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        # text: [batch size, seq len]
        embedded = self.embedding(text)
        # embedded: [batch size, seq len, embedding dim]
        
        output, (hidden, cell) = self.lstm(embedded)
        # output: [batch size, seq len, hidden dim]
        # hidden: [1, batch size, hidden dim]

        # We only use the last hidden state
        hidden = hidden.squeeze(0)

        output = self.fc(hidden)
        # output: [batch size, output dim]

        return output