import torch
import torch.nn as nn
# from pytorch_transformers import *
from transformers import *

class CLS_model(nn.Module):
    def __init__(self, config, num_labels=2):
        super(CLS_model, self).__init__()
        # Load pre-trained  model

        self.config = config

        if "mnli" in config.pretrained_weight.lower():
            self.model = AutoModel.from_pretrained(config.pretrained_weight, num_labels=num_labels)
            self.linear = nn.Linear(768, num_labels)

        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(config.pretrained_weight, num_labels=num_labels)

    def forward(self, x):
        if "mnli" in self.config.pretrained_weight.lower():
            cls_rep = self.model(x, x>0)[1]
            return self.linear(cls_rep)
        else:
            # Encode input text
            output = self.model(x, x>0)

            return output[0]
