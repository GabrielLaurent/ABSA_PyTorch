# src/data/dataset.py

import torch
from torch.utils.data import Dataset

class ABSADataset(Dataset):
    """Dataset for aspect-based sentiment analysis.

    This dataset class prepares the data for training and evaluation.
    It tokenizes the input text and returns the input IDs, attention mask,
    and sentiment label.
    """

    def __init__(self, data, tokenizer, max_len):
        """Initializes the ABSADataset.

        Args:
            data (list): List of dictionaries containing 'text', 'aspect', and 'sentiment'.
            tokenizer (transformers.PreTrainedTokenizer): Tokenizer to use for encoding the data.
            max_len (int): Maximum length of the input sequences.
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_map = {"negative": 0, "neutral": 1, "positive": 2}

    def __len__(self):
        """Returns the length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """Returns a single item from the dataset.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            dict: Dictionary containing the input IDs, attention mask, and sentiment label.
        """
        item = self.data[idx]
        text = item['text']
        aspect = item['aspect']
        sentiment = item['sentiment']

        # Combine text and aspect for input
        combined_text = f"{text} {aspect}"

        # Tokenize the combined text
        encoding = self.tokenizer(combined_text,
                                 add_special_tokens=True,
                                 max_length=self.max_len,
                                 truncation=True,
                                 padding='max_length',
                                 return_attention_mask=True,
                                 return_tensors='pt')

        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()
        labels = torch.tensor(self.label_map[sentiment], dtype=torch.long)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
