import torch
import torch.optim as optim
from src.data.dataloaders import create_dataloaders
from src.models.model import get_model
from src.training.trainer import train
from src.utils.config import Config


def main():
    # Load configuration
    config = Config('config.json')

    # Create dataloaders
    train_loader, val_loader, test_loader, vocabulary = create_dataloaders(
        config.data_dir,
        config.batch_size,
        config.max_len,
        config.word2vec_file
    )

    # Model parameters
    vocab_size = len(vocabulary)
    embedding_dim = config.embedding_dim
    hidden_dim = config.hidden_dim
    output_dim = config.output_dim
    num_layers = config.num_layers
    dropout = config.dropout

    # Load pre-trained embeddings
    embedding_weights = vocabulary.vectors

    # Initialize the model
    model = get_model(
        config.model_name,
        vocab_size,
        embedding_dim,
        hidden_dim,
        output_dim,
        embedding_weights,
        num_layers,
        dropout
    )

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Define loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Move model and criterion to device (GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion.to(device)

    # Train the model
    train(model, train_loader, val_loader, criterion, optimizer, device, config.epochs, config.checkpoint_dir)

if __name__ == '__main__':
    main()