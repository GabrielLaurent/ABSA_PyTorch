import torch
import torch.nn as nn
import torch.nn.functional as F

class ABSAModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, embedding_weights):
        super(ABSAModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_weights, dtype=torch.float32))
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim) # Bi-directional LSTM so * 2

    def forward(self, text, aspect, text_lengths, aspect_lengths):
        # text: [batch size, seq len]
        # aspect: [batch size, aspect len]
        # text_lengths: [batch size]
        # aspect_lengths: [batch size]

        # embedded: [batch size, seq len, emb dim]
        embedded_text = self.embedding(text)
        embedded_aspect = self.embedding(aspect)

        # Pack padded sequences
        packed_embedded_text = nn.utils.rnn.pack_padded_sequence(embedded_text, text_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_embedded_aspect = nn.utils.rnn.pack_padded_sequence(embedded_aspect, aspect_lengths.cpu(), batch_first=True, enforce_sorted=False)

        # lstm_out: [batch size, seq len, hid dim * 2]
        lstm_out_text, (hidden_text, cell_text) = self.lstm(packed_embedded_text)
        lstm_out_aspect, (hidden_aspect, cell_aspect) = self.lstm(packed_embedded_aspect)

        # Unpack
lstm_out_text, _ = nn.utils.rnn.pad_packed_sequence(lstm_out_text, batch_first=True)
lstm_out_aspect, _ = nn.utils.rnn.pad_packed_sequence(lstm_out_aspect, batch_first=True)

        # Average the hidden states for aspect representation
aspect_representation = torch.sum(lstm_out_aspect, dim=1) / aspect_lengths.unsqueeze(1).float().to(lstm_out_aspect.device)

        # Concatenate text and aspect representations (you might want to implement attention here)
        # Here, simply using the last hidden state from the text LSTM
        # last_hidden_text = hidden_text[-1,:,:]
        # concatenated = torch.cat((last_hidden_text, aspect_representation), dim=1) # dimension error due to bidirectional lstm
        # Alternatively, take the last hidden state from both directions:
        last_hidden_forward = hidden_text[0,:,:]  # Forward direction
        last_hidden_backward = hidden_text[1,:,:] # Backward direction
        concatenated = torch.cat((last_hidden_forward, last_hidden_backward, aspect_representation), dim=1)
        concatenated = torch.relu(concatenated) #add relu to stabilize
        # Fully connected layer
        output = self.fc(concatenated)

        return output