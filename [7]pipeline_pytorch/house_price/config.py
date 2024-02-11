import torch
import torch.nn.functional as F
import torchmetrics
from nn import ANN

config = {

  'files': {
    'X_csv': './trn_X.csv',
    'y_csv': './trn_y.csv',
    'test_X_csv': './trn_X.csv',
    'test_y_csv': './trn_y.csv',
    'output': './model.pth',
    'output_csv': './results/five_fold.csv',
  },

  'model': ANN,
  'model_params': {
    'input_dim': 'auto', # Always will be determined by the data shape
    'hidden_dim': 128,
    'dropout': 0.3,
    'activation': F.relu,
  },

  'train_params': {
    'data_loader_params': {
      'batch_size': 32,
      'shuffle': True,
    },
    'loss': F.mse_loss,
    'optim': torch.optim.Adam,
    'optim_params': {
      'lr': 0.01,
    },
    'metric': torchmetrics.MeanSquaredError(squared=False),
    'device': 'cpu',
    'epochs': 50,
  },

  'cv_params':{
    'n_split': 5,
  },

}