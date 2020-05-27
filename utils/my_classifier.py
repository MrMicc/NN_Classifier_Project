import torch
from torch import nn
import torch.nn.functional as F


class Classifier(nn.Module):

    def __init__(self, input_w, output_w, hidden_layers, drop_out=0.2):
        super().__init__()
        self.input_w = input_w
        self.output_w = output_w

        layers = zip(hidden_layers[:-1], hidden_layers[1:])

        self.hidden_layers = nn.ModuleList([nn.Linear(input_w, hidden_layers[0])])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layers])

        self.output = nn.Linear(hidden_layers[-1], output_w)

        self.dropout = nn.Dropout(p=drop_out)

    def forward(self, x):
        x = x.view(x.shape[0], -1)

        for layer in self.hidden_layers:
            x = self.dropout(F.relu(layer(x)))

        return F.log_softmax(self.output(x), dim=1)
