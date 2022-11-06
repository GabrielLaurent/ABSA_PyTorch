import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
import logging

class Trainer:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.logger = logging.getLogger(__name__)

    def train_epoch(self, data_loader):
        self.model.train()
        total_loss = 0
        for batch in data_loader:
            inputs = batch['text'].to(self.device)
            labels = batch['label'].to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(data_loader)

    def evaluate(self, data_loader):
        self.model.eval()
        true_labels = []
        predicted_labels = []
        total_loss = 0
        with torch.no_grad():
            for batch in data_loader:
                inputs = batch['text'].to(self.device)
                labels = batch['label'].to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                true_labels.extend(labels.cpu().numpy())
                predicted_labels.extend(predicted.cpu().numpy())

        accuracy = accuracy_score(true_labels, predicted_labels)
        f1 = f1_score(true_labels, predicted_labels, average='macro') # or 'binary', 'micro', 'weighted'
        avg_loss = total_loss / len(data_loader)

        self.logger.info(f"Evaluation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        return avg_loss, accuracy, f1

    def predict(self, data_loader):
      self.model.eval()
      predicted_labels = []
      with torch.no_grad():
        for batch in data_loader:
          inputs = batch['text'].to(self.device)
          outputs = self.model(inputs)
          _, predicted = torch.max(outputs, 1)
          predicted_labels.extend(predicted.cpu().numpy())
      return predicted_labels