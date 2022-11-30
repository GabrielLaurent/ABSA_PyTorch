import json
from torch.utils.data import DataLoader
from src.data.dataset import ABSADataset
from src.data.vocabulary import Vocabulary
import nltk

def create_data_loaders(train_file, val_file, batch_size=32):
    # Load data from JSON files
    with open(train_file, 'r') as f:
        train_data = json.load(f)

    with open(val_file, 'r') as f:
        val_data = json.load(f)

    # Build vocabulary from training data
    all_tokens = []
    tokenizer = nltk.word_tokenize
    for item in train_data:
        all_tokens.extend(tokenizer(item['text']))
        all_tokens.extend(tokenizer(item['aspect']))
    vocabulary = Vocabulary(set(all_tokens))

    # Create datasets
    train_dataset = ABSADataset(train_data, vocabulary)
    val_dataset = ABSADataset(val_data, vocabulary)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, vocabulary