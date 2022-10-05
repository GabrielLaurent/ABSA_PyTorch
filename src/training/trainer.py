# src/training/trainer.py

import torch
import torch.nn as nn
from tqdm import tqdm

def train_epoch(model, data_loader, optimizer, device):
    """Trains the model for one epoch.

    Args:
        model (nn.Module): Model to train.
        data_loader (torch.utils.data.DataLoader): Data loader for training data.
        optimizer (torch.optim.Optimizer): Optimizer to use for training.
        device (torch.device): Device to train on.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    total_loss = 0
    criterion = nn.CrossEntropyLoss()

    for batch in tqdm(data_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)


def evaluate(model, data_loader, device):
    """Evaluates the model on the given data.

    Args:
        model (nn.Module): Model to evaluate.
        data_loader (torch.utils.data.DataLoader): Data loader for evaluation data.
        device (torch.device): Device to evaluate on.

    Returns:
        tuple: Tuple containing the average loss and accuracy.
    """
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted_labels = torch.max(outputs, 1)
            correct_predictions += (predicted_labels == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy
