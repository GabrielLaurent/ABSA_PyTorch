import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, embedding_weights, num_layers=1, dropout=0.0):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_weights))
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=True, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # * 2 for bidirectional
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.embedding(text)
        output, (hidden, cell) = self.lstm(embedded)
        output = self.dropout(output[:, -1, :]) # Take the last time step's output
        return self.fc(output)


class GRUModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, embedding_weights, num_layers=1, dropout=0.0):
        super(GRUModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_weights))
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=True, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # * 2 for bidirectional
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.embedding(text)
        output, hidden = self.gru(embedded)
        output = self.dropout(output[:, -1, :])  # Take the last time step's output
        return self.fc(output)


class AttentionModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, embedding_weights):
        super(AttentionModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_weights))
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.attention = Attention(hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        lstm_out, _ = self.lstm(embedded)
        attention_out = self.attention(lstm_out)
        return self.fc(attention_out)

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention_weights = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, lstm_output):
        # lstm_output: (batch_size, seq_len, hidden_dim)
        attention_logits = torch.matmul(lstm_output, self.attention_weights)
        # attention_logits: (batch_size, seq_len)
        attention_weights = F.softmax(attention_logits, dim=1)
        # attention_weights: (batch_size, seq_len)
        attention_output = torch.sum(lstm_output * attention_weights.unsqueeze(-1), dim=1)
        # attention_output: (batch_size, hidden_dim)
        return attention_output




def get_model(model_name, vocab_size, embedding_dim, hidden_dim, output_dim, embedding_weights, num_layers=1, dropout=0.0):
    if model_name == 'lstm':
        return LSTMModel(vocab_size, embedding_dim, hidden_dim, output_dim, embedding_weights, num_layers, dropout)
    elif model_name == 'gru':
        return GRUModel(vocab_size, embedding_dim, hidden_dim, output_dim, embedding_weights, num_layers, dropout)
    elif model_name == 'attention':
        return AttentionModel(vocab_size, embedding_dim, hidden_dim, output_dim, embedding_weights)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")