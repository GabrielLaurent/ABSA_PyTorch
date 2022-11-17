import torch
import json
from src.models.model import AspectBasedSentimentAnalysis
from src.data.dataloaders import create_dataloaders
from src.training.trainer import Trainer
from src.utils.config import Config


def main():
    config = Config('config.json')
    config = config.get_config()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = AspectBasedSentimentAnalysis(config['embedding_dim'], config['hidden_dim'], config['num_classes']).to(device)
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(config['data_path'], config['batch_size'])

    trainer = Trainer(model, config, device)

    for epoch in range(config['epochs']):
        train_loss = trainer.train_epoch(train_dataloader)
        print(f"Epoch {epoch+1}/{config['epochs']}, Train Loss: {train_loss:.4f}")

        # Add validation and testing later

if __name__ == "__main__":
    main()