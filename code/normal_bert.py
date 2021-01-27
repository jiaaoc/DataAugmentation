import torch
import torch.nn as nn
# from pytorch_transformers import *
from transformers import *

class ClassificationBert(nn.Module):
    def __init__(self, num_labels=2):
        super(ClassificationBert, self).__init__()
        # Load pre-trained bert model
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.linear = nn.Sequential(nn.Linear(768, num_labels))
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, length=256):
        # Encode input text
        output = self.bert(x)
        all_hidden = output[0]

        # input_length = torch.sum(x > 0, dim=1)
        # input_length = torch.max(input_length, torch.ones_like(input_length))
        #
        # pooled_output = torch.sum(all_hidden, 1) / input_length[:,None]

        cls_rep = all_hidden[:,0,:]
        pooled_output = self.dropout(cls_rep)
        # Use linear layer to do the predictions
        predict = self.linear(pooled_output)

        return predict
