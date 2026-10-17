"""Shared residual projection for cross-scene adaptation."""
from collections import OrderedDict

import torch.nn as nn

class Projector(nn.Module):
    """Add an identity residual to the batch-normalized projection branch."""

    def __init__(self, input_dim=512, output_dim=512, hidden_dim=256):
        super(Projector, self).__init__()
        if input_dim != output_dim:
            raise ValueError("Residual projection requires matching input and output dimensions")
        self.output_dim = output_dim
        # Keep the identity path outside both dropout operations.
        self.projection = nn.Sequential(OrderedDict([
            ("0", nn.Linear(input_dim, hidden_dim)),
            ("1", nn.BatchNorm1d(hidden_dim)),
            ("2", nn.ReLU(inplace=True)),
            ("dropout_after_relu", nn.Dropout(0.01)),
            ("3", nn.Linear(hidden_dim, output_dim)),
            ("4", nn.BatchNorm1d(output_dim)),
            ("dropout_after_bn", nn.Dropout(0.01)),
        ]))
        nn.init.constant_(self.projection._modules["4"].weight, 0.01)

    def forward(self, x):
        return self.projection(x) + x
