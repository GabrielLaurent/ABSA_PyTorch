import torch
import torch.nn as nn
import torch.optim as optim
from src.models.model import SentimentModel
from src.data.dataloaders import create_data_loaders
from src.utils.config import Config
import json
import random
import os
from src.training.trainer import Trainer


def main():
    # Load Configuration
    config = Config('config.json')
    params = config.get_config()

    # Set Random Seed for Reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    random.seed(42)

    # Load Data
    train_loader, val_loader, test_loader, vocab = create_data_loaders(
        params['data']['train_file'], params['data']['val_file'], params['data']['test_file'], params['data']['vocab_file'], params['training']['batch_size']
    )

    # Training Loop with Hyperparameter Tuning
    best_val_loss = float('inf')
    best_model_state = None  # Store the state of the best model
    best_hyperparameters = None

    # Hyperparameter Tuning
    tuning_params = params['tuning']
    learning_rates = tuning_params.get('learning_rate', [config.get('training.learning_rate')])
    batch_sizes = tuning_params.get('batch_size', [config.get('training.batch_size')])
    embedding_dims = tuning_params.get('embedding_dim', [config.get('model.embedding_dim')])
    hidden_dims = tuning_params.get('hidden_dim', [config.get('model.hidden_dim')])


    # Grid Search (modify this loop for random search or other techniques)
    for lr in learning_rates:
        for batch_size in batch_sizes:
            for embedding_dim in embedding_dims:
                for hidden_dim in hidden_dims:
                    print(f"\n##### Tuning with: lr={lr}, batch_size={batch_size}, embedding_dim={embedding_dim}, hidden_dim={hidden_dim} #####")
                    
                    # Update DataLoaders
                    train_loader, val_loader, test_loader, vocab = create_data_loaders(
                        params['data']['train_file'], params['data']['val_file'], params['data']['test_file'], params['data']['vocab_file'], batch_size
                    )

                    # Update Model and Optimizer
                    model = SentimentModel(len(vocab), embedding_dim, hidden_dim, config.get('model.dropout'))
                    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=config.get('training.weight_decay'))
                    criterion = nn.CrossEntropyLoss()

                    if torch.cuda.is_available():
                        model = model.cuda()
                        criterion = criterion.cuda()

                    # Trainer
                    trainer = Trainer(model, optimizer, criterion, train_loader, val_loader, config.get('training.clip_grad'))

                    # Training
                    for epoch in range(config.get('training.num_epochs')):
                        train_loss, train_acc = trainer.train_epoch()
                        val_loss, val_acc = trainer.evaluate()
                        print(f"Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

                    # Evaluate on Validation
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model_state = model.state_dict()
                        best_hyperparameters = {'learning_rate': lr, 'batch_size': batch_size, 'embedding_dim': embedding_dim, 'hidden_dim': hidden_dim}
                        print(f"New best validation loss: {best_val_loss:.4f} with parameters: {best_hyperparameters}")


    # Load best model after tuning is completed
    print(f"\n##### Best Hyperparameters: {best_hyperparameters} #####")
    model = SentimentModel(len(vocab), best_hyperparameters['embedding_dim'], best_hyperparameters['hidden_dim'], config.get('model.dropout'))
    model.load_state_dict(best_model_state)

    if torch.cuda.is_available():
      model = model.cuda()

    # Perform final evaluation with best model
    trainer = Trainer(model, optimizer, criterion, train_loader, val_loader, config.get('training.clip_grad')) # Using last seen train_loader but should be fine as best model has been loaded
    test_loss, test_acc = trainer.evaluate(test_loader)

    print(f"\n##### FINAL TEST RESULTS #####")
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

if __name__ == "__main__":
    main()
