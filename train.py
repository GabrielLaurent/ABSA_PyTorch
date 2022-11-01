import torch
import torch.nn as nn
import torch.optim as optim
from src.data.dataloaders import create_dataloaders
from src.models.model import ABSAClassifier
from src.utils.config import Config
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train(model, train_loader, val_loader, optimizer, criterion, config):
    logging.info("Starting training...")
    for epoch in range(config.num_epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            text, aspect, label = batch
            outputs = model(text, aspect)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Evaluate on validation set
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                text, aspect, label = batch
                outputs = model(text, aspect)
                loss = criterion(outputs, label)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

        val_loss /= len(val_loader)
        accuracy = 100 * correct / total

        logging.info(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {accuracy:.2f}%')

        # Check for overfitting/underfitting (basic check)
        if epoch > 0 and val_loss > train_loss:
            logging.warning("Possible overfitting detected.")

    logging.info("Training finished!")

if __name__ == '__main__':
    config = Config("config.json")
    train_loader, val_loader, _ = create_dataloaders(config)

    model = ABSAClassifier(config.vocab_size, config.embedding_dim, config.hidden_dim, config.num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    train(model, train_loader, val_loader, optimizer, criterion, config)