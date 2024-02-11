import torch.nn as nn
import torch.nn.functional as F

class ANN(nn.Module):
  def __init__(self, input_dim:int=5, hidden_dim:int=128, dropout:float=0.3, activation=F.relu):
    super().__init__()
    self.lin1 = nn.Linear(input_dim,hidden_dim)
    self.lin2 = nn.Linear(hidden_dim,1)
    self.dropout = nn.Dropout(dropout)
    self.activation = activation
  def forward(self, x):
    x = self.lin1(x)
    x = self.activation(x)
    x = self.dropout(x)
    x = self.lin2(x)
    return x