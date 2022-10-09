import torch
import torch.nn as nn
import os

class Trainer:
    def __init__(self, model, optimizer, criterion, config, device, train_dataloader, val_dataloader):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.config = config
        self.device = device
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.best_val_loss = float('inf')
        self.checkpoint_dir = config['training']['checkpoint_dir'] # Read checkpoint directory from config

        # Create checkpoint directory if it doesn't exist
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    def train_epoch(self):
        self.model.train()
        total_loss = 0

        for batch in self.train_dataloader:
            inputs = batch['text'].to(self.device)
            aspects = batch['aspect'].to(self.device)
            labels = batch['label'].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs, aspects)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_dataloader)

    def validate_epoch(self):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in self.val_dataloader:
                inputs = batch['text'].to(self.device)
                aspects = batch['aspect'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(inputs, aspects)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()

        return total_loss / len(self.val_dataloader)

    def train(self):
        num_epochs = self.config['training']['epochs']
        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate_epoch()

            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

            # Save checkpoint if validation loss improved
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss)
                print(f'\t--> Checkpoint saved - val_loss: {val_loss:.4f}')

    def save_checkpoint(self, epoch, val_loss):
        checkpoint_path = os.path.join(self.checkpoint_dir, 'best_model.pth') # Save as best_model.pth by default, overwriting previous best
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
        }, checkpoint_path)
