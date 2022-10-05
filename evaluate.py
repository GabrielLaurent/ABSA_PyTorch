# evaluate.py

import torch
from src.models.model import ABSAModel
from src.data.dataloaders import create_data_loader
from src.utils.config import Config
from src.training.trainer import evaluate


def main():
    """Main evaluation function.
    """
    # Load configuration
    config = Config('config.json')

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load tokenizer and model
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.bert_model_name)
    model = ABSAModel(config.bert_model_name, config.num_classes).to(device)

    # Load the checkpoint
    model.load_state_dict(torch.load(config.checkpoint_path))
    model.eval()

    # Prepare data loader
    # Replace the following sample data with your evaluation data.
    eval_data = [
        {"text": "The service was slow.", "aspect": "service", "sentiment": "negative"},
        {"text": "I enjoyed the music.", "aspect": "music", "sentiment": "positive"},
    ]

    eval_loader = create_data_loader(eval_data, tokenizer, config.max_len, config.batch_size, shuffle=False)

    # Evaluate the model
    eval_loss, eval_accuracy = evaluate(model, eval_loader, device)

    print(f"Evaluation Loss: {eval_loss:.4f}, Evaluation Accuracy: {eval_accuracy:.4f}")


if __name__ == '__main__':
    main()
