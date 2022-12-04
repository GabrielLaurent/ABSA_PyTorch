import torch
from torch.utils.data import DataLoader


def create_data_loaders(train_dataset, val_dataset, batch_size, shuffle=True, num_workers=0):
    """Creates PyTorch DataLoaders for training and validation datasets.

    Args:
        train_dataset (Dataset): Training dataset.
        val_dataset (Dataset): Validation dataset.
        batch_size (int): Batch size.
        shuffle (bool, optional): Whether to shuffle the training data. Defaults to True.
        num_workers (int, optional): Number of worker processes for data loading. Defaults to 0.

    Returns:
        tuple: A tuple containing the training and validation DataLoaders.
    """

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle validation data
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


if __name__ == '__main__':
    # Example usage (replace with your actual dataset and parameters)
    from src.data.dataset import ABSADataset  # Import your Dataset class
    from src.data.vocabulary import Vocabulary
    import json

    #Dummy Vocabulary and Config
    vocab = Vocabulary()
    config = {
        "max_seq_length": 100
    }
    with open('data/train.json', 'w') as f:
        json.dump([{"text": "This movie is great, I loved it", "aspect": "movie", "sentiment": "positive"}], f)
    
    train_dataset = ABSADataset("data/train.json", vocab, config)
    val_dataset = ABSADataset("data/train.json", vocab, config)

    train_loader, val_loader = create_data_loaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0
    )

    # Iterate through the training data (example)
    for batch in train_loader:
        inputs, aspects, labels = batch
        print("Input batch shape:", inputs.shape)
        print("Aspect batch shape:", aspects.shape)
        print("Label batch shape:", labels.shape)
        break  # Only print the first batch
