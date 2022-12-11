# src/training/trainer.py

import torch
import torch.nn as nn
import torch.optim as optim

def train(model, data_loader, optimizer, criterion, epochs, device):
    model.train()
    for epoch in range(epochs):
        for batch in data_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            sentiments = batch['sentiment'].to(device)
            outputs = model(input_ids)
            loss = criterion(outputs, sentiments)
            loss.backward()
            optimizer.step()
        print(f'Epoch {{epoch+1}}, Loss: {{loss.item():.4f}}')

def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            sentiments = batch['sentiment'].to(device)
            outputs = model(input_ids)
            loss = criterion(outputs, sentiments)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == sentiments).sum().item()
    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions / len(data_loader.dataset)
    return avg_loss, accuracy