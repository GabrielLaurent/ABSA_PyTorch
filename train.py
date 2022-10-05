# train.py

import torch
import torch.optim as optim
from src.models.model import ABSAModel
from src.data.dataloaders import create_data_loader
from src.utils.config import Config
from src.training.trainer import train_epoch, evaluate
import os


def main():
    """Main training loop.
    """
    # Load configuration
    config = Config('config.json')

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load tokenizer and model
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.bert_model_name)
    model = ABSAModel(config.bert_model_name, config.num_classes).to(device)

    # Prepare data loaders
    # Load your data here. Replace the sample data with your actual data.
    train_data = [
        {"text": "This movie is great!", "aspect": "movie", "sentiment": "positive"},
        {"text": "The food was terrible.", "aspect": "food", "sentiment": "negative"},
    ]
    val_data = [
        {"text": "The service was okay.", "aspect": "service", "sentiment": "neutral"},
        {"text": "I loved the atmosphere.", "aspect": "atmosphere", "sentiment": "positive"},
    ]

    train_loader = create_data_loader(train_data, tokenizer, config.max_len, config.batch_size)
    val_loader = create_data_loader(val_data, tokenizer, config.max_len, config.batch_size, shuffle=False)

    # Define optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)

    # Training loop
    for epoch in range(config.epochs):
        print(f"Epoch {epoch + 1}/{config.epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_accuracy = evaluate(model, val_loader, device)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        # Save checkpoint
        if not os.path.exists('checkpoints'):
            os.makedirs('checkpoints')
        torch.save(model.state_dict(), f'checkpoints/model_epoch_{epoch + 1}.pth')

if __name__ == '__main__':
    main()
