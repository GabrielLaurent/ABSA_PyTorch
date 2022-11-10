import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

class Trainer:
    def __init__(self, model: nn.Module, train_dataloader: DataLoader, val_dataloader: DataLoader, config: dict):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.device = config['device']
        self.model.to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

    def train_epoch(self, epoch_num: int):
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader), desc=f"Epoch {epoch_num}")
        for idx, batch in progress_bar:
            self.optimizer.zero_grad()

            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            outputs = self.model(input_ids, attention_mask)
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(outputs, labels)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            progress_bar.set_postfix({"loss": total_loss / (idx + 1)})

        return total_loss / len(self.train_dataloader)

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in self.val_dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids, attention_mask)
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(outputs, labels)

                total_loss += loss.item()

        return total_loss / len(self.val_dataloader)

    def train(self):
        for epoch in range(self.config['epochs']):
            train_loss = self.train_epoch(epoch + 1)
            val_loss = self.evaluate()
            print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
