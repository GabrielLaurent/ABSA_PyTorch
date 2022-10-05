# src/data/dataloaders.py

import torch
from torch.utils.data import DataLoader
from src.data.dataset import ABSADataset


def create_data_loader(data, tokenizer, max_len, batch_size, shuffle=True):
    """Creates a data loader for the given data.

    Args:
        data (list): List of data samples.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer to use for encoding the data.
        max_len (int): Maximum length of the input sequences.
        batch_size (int): Batch size for the data loader.
        shuffle (bool): Whether to shuffle the data.

    Returns:
        torch.utils.data.DataLoader: Data loader for the given data.
    """
    dataset = ABSADataset(data, tokenizer, max_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

if __name__ == '__main__':
    # Example Usage (replace with actual data and tokenizer initialization)
    # This is just for demonstration purposes
    dummy_data = [
        {"text": "This movie is great!", "aspect": "movie", "sentiment": "positive"},
        {"text": "The food was terrible.", "aspect": "food", "sentiment": "negative"},
    ]
    
    class DummyTokenizer:
        def __init__(self):
            self.vocab = {"<PAD>": 0, "this": 1, "movie": 2, "is": 3, "great": 4, "food": 5, "was": 6, "terrible": 7, "!": 8, ".": 9}
            self.pad_token_id = 0
            
        def tokenize(self, text):
            return text.lower().split()

        def convert_tokens_to_ids(self, tokens):
            return [self.vocab.get(token, 0) for token in tokens]

        def __call__(self, text, padding=True, truncation=True, max_length=10, return_tensors='pt'):
            tokens = self.tokenize(text)
            ids = self.convert_tokens_to_ids(tokens)

            if padding:
                ids = ids + [self.pad_token_id] * (max_length - len(ids))
            ids = ids[:max_length] #Truncate if necessary

            return {'input_ids': torch.tensor([ids])}


    dummy_tokenizer = DummyTokenizer()
    max_length = 10
    batch_size = 2

    data_loader = create_data_loader(dummy_data, dummy_tokenizer, max_length, batch_size)

    for batch in data_loader:
        print(batch)
        break  # Print only one batch for demonstration