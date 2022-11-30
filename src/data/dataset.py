import torch
from torch.utils.data import Dataset
import nltk
# nltk.download('punkt')  # Download punkt tokenizer if you haven't already

class ABSADataset(Dataset):
    def __init__(self, data, vocabulary):
        self.data = data
        self.vocabulary = vocabulary
        self.tokenizer = nltk.word_tokenize  # Use nltk tokenizer

        self.processed_data = self._preprocess_data()
    
    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        return self.processed_data[idx]

    def _preprocess_data(self):
        processed_data = []
        for item in self.data:
            text = item['text']
            aspect = item['aspect']
            sentiment = item['sentiment']

            # Tokenize text and aspect
            text_tokens = self.tokenizer(text)
            aspect_tokens = self.tokenizer(aspect)

            # Map tokens to indices using the vocabulary
            text_indices = [self.vocabulary.token_to_index(token) for token in text_tokens]
            aspect_indices = [self.vocabulary.token_to_index(token) for token in aspect_tokens]

            # Convert sentiment to numerical label (assuming a mapping like positive: 2, negative: 0, neutral: 1)
            sentiment_label = {'positive': 2, 'negative': 0, 'neutral': 1}[sentiment]

            processed_data.append({
                'text': torch.tensor(text_indices),
                'aspect': torch.tensor(aspect_indices),
                'sentiment': torch.tensor(sentiment_label)
            })
        return processed_data