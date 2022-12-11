# src/data/dataloaders.py

from torch.utils.data import DataLoader
from src.data.dataset import ABSADataset  # Import ABSADataset

def create_data_loader(data, tokenizer, max_len, batch_size):
    dataset = ABSADataset(data, tokenizer, max_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)