import torch
import torch.nn.functional as F
import os
import subprocess
import sys

# Install PyG and its dependencies (only if not already installed)
try:
    import torch_geometric
except ImportError:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "torch-scatter==2.1.2",
        "torch-sparse==0.6.17",
        "torch-cluster==1.6.1",
        "torch-spline-conv==1.2.2",
        "torch-geometric==2.5.2"
    ])

from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)  # Suitable for NLLLoss
