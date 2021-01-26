import torch
import torch.nn as nn
# from pytorch_transformers import *
from transformers import *

class CLS_model(nn.Module):
    def __init__(self, config, num_labels=2):
        super(ClassificationBert, self).__init__()
        # Load pre-trained  model
        self.model = AutoModelForSequenceClassification.from_pretrained(config.pretrained_weight, num_labels=num_labels)

    def forward(self, x):
        # Encode input text
        output = self.model(x, x>0)
        return output
