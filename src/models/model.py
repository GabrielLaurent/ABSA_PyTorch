# src/models/model.py

import torch
import torch.nn as nn
from transformers import BertModel

class ABSAModel(nn.Module):
    """Aspect-Based Sentiment Analysis Model.

    This model uses a pre-trained BERT model to encode the input text and aspect,
    and then uses a linear layer to predict the sentiment.
    """
    def __init__(self, bert_model_name, num_classes):
        """Initializes the ABSAModel.

        Args:
            bert_model_name (str): Name of the pre-trained BERT model to use.
            num_classes (int): Number of sentiment classes.
        """
        super(ABSAModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        """Performs a forward pass through the model.

        Args:
            input_ids (torch.Tensor): Input IDs.
            attention_mask (torch.Tensor): Attention mask.

        Returns:
            torch.Tensor: Sentiment predictions.
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.linear(pooled_output)
        return logits
