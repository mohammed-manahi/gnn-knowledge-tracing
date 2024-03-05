import torch
from torch import nn
from torch_geometric.nn import SAGEConv
from constants import *


class GraphSAGE(torch.nn.Module):
    def __init__(self, num_skills, num_layers=3, hidden_dim=skill_embd_dim):
        """
        Represents a GraphSAGE graph neural network with num_layers
        layers and embedding and hidden dimension of hidden_dim.
        """
        super().__init__()
        self.tag = 'GraphSAGE'
        self.pre_embs = nn.Embedding(num_skills, hidden_dim)
        self.conv1 = SAGEConv(hidden_dim, hidden_dim)
        self.prelu1 = nn.PReLU()
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.prelu2 = nn.PReLU()
        self.conv3 = SAGEConv(hidden_dim, hidden_dim)
        self.prelu3 = nn.PReLU()
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index, edge_weight):
        """
        Runs a forward pass through GraphSAGE with given initial skill IDs and
        edge_index and edge_weights.

        Arguments:
          - x: skill IDs (torch.Tensor)
          - edge_index: edges in skill graph (torch.Tensor)
          - edge_weight: edge weights of skill graph (torch.Tensor)

        Returns:
          - final node embedding for skill
        """
        h0 = self.pre_embs(x)  # initial skill embedding
        h1 = self.dropout(self.prelu1(self.conv1(h0, edge_index)))
        h2 = self.dropout(self.prelu2(self.conv2(h1, edge_index)))
        h3 = self.prelu3(self.conv3(h2, edge_index))
        return self.out(h3)
