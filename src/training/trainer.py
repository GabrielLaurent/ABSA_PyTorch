import torch
import torch.nn as nn
import torch.optim as optim

class Trainer:
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        for batch in dataloader:
            inputs = batch['text'].to(self.device)
            aspects = batch['aspect'].to(self.device)
            targets = batch['sentiment'].to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(inputs, aspects)
            loss = self.loss_function(outputs, targets)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def evaluate(self, dataloader):
        self.model.eval()
        # Add evaluation logic later
        pass