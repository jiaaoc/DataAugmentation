import torch
import torch.nn as nn
# from pytorch_transformers import *
from transformers import *

class ClassificationBert(nn.Module):
    def __init__(self, num_labels=2):
        super(ClassificationBert, self).__init__()
        # Load pre-trained bert model
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.linear = nn.Sequential(nn.Linear(768, 128),
                                    nn.Tanh(),
                                    nn.Linear(128, num_labels))

    def forward(self, x, length=256):
        # Encode input text
        output = self.bert(x)
        all_hidden = output[0]

        pooled_output = torch.sum(all_hidden, 1) / torch.sum(x > 0, dim=1)[:,None]

        # Use linear layer to do the predictions
        predict = self.linear(pooled_output)

        return predict
