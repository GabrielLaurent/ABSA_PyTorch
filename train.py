# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from src.models.model import ABSAModel
from src.data.dataloaders import create_data_loader
from src.training.trainer import train, evaluate
from src.utils.config import load_config
from transformers import AutoTokenizer

if __name__ == '__main__':
    # Load configuration
    config = load_config('config.json')

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data (example data)
    train_data = [
        {'text': 'The food was great', 'aspect': 'food', 'sentiment': 1},
        {'text': 'The service was bad', 'aspect': 'service', 'sentiment': 0}
    ]
    test_data = [
        {'text': 'The price was high', 'aspect': 'price', 'sentiment': 0},
        {'text': 'The atmosphere was nice', 'aspect': 'atmosphere', 'sentiment': 1}
    ]

    tokenizer = AutoTokenizer.from_pretrained(config['pretrained_model'])
    # Create data loaders
    train_loader = create_data_loader(train_data, tokenizer, config['max_len'], config['batch_size'])
    test_loader = create_data_loader(test_data, tokenizer, config['max_len'], config['batch_size'])

    # Initialize model
    model = ABSAModel(len(tokenizer), config['embedding_dim'], config['hidden_dim'], config['output_dim']).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    # Train the model
    train(model, train_loader, optimizer, criterion, config['epochs'], device)

    # Evaluate the model
    loss, accuracy = evaluate(model, test_loader, criterion, device)
    print(f'Test Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')