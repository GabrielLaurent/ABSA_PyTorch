# evaluate.py

import torch
import torch.nn as nn
from src.models.model import ABSAModel
from src.data.dataloaders import create_data_loader
from src.training.trainer import evaluate
from src.utils.config import load_config
from transformers import AutoTokenizer

if __name__ == '__main__':
    # Load configuration
    config = load_config('config.json')

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data (example data)
    test_data = [
        {'text': 'The price was high', 'aspect': 'price', 'sentiment': 0},
        {'text': 'The atmosphere was nice', 'aspect': 'atmosphere', 'sentiment': 1}
    ]

    tokenizer = AutoTokenizer.from_pretrained(config['pretrained_model'])

    # Create data loaders
    test_loader = create_data_loader(test_data, tokenizer, config['max_len'], config['batch_size'])

    # Initialize model
    model = ABSAModel(len(tokenizer), config['embedding_dim'], config['hidden_dim'], config['output_dim']).to(device)

    # Load the model checkpoint
    model.load_state_dict(torch.load('checkpoints/model.pth'))  # Replace with your checkpoint path
    model.eval()

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Evaluate the model
    loss, accuracy = evaluate(model, test_loader, criterion, device)
    print(f'Test Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')