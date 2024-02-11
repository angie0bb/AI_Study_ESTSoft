import torch
from torch import nn
from torch.utils.data import DataLoader
import torchmetrics
from typing import Optional

def train_one_epoch(
  model:nn.Module,
  criterion:callable,
  optimizer:torch.optim.Optimizer,
  data_loader:DataLoader,
  device:str
) -> float:
  '''train one epoch
  
  Args:
      model: model
      criterion: loss
      optimizer: optimizer
      data_loader: data loader
      device: device
  '''
  model.train()
  total_loss = 0.
  for X, y in data_loader:
    X, y = X.to(device), y.to(device)
    output = model(X)
    loss = criterion(output, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    total_loss += loss.item() * len(y)
  return total_loss/len(data_loader.dataset)

def evaluate(
  model:nn.Module,
  criterion:callable,
  data_loader:DataLoader,
  device:str,
  metric:Optional[torchmetrics.metric.Metric]=None,
) -> float:
  '''evaluate
  
  Args:
      model: model
      criterions: list of criterion functions
      data_loader: data loader
      device: device
  '''
  model.eval()
  total_loss = 0.
  with torch.inference_mode():
    for X, y in data_loader:
      X, y = X.to(device), y.to(device)
      output = model(X)
      total_loss += criterion(output, y).item() * len(y)
      if metric is not None:
        metric.update(output, y)
  return total_loss/len(data_loader.dataset)


def main(args):
  import numpy as np
  import pandas as pd
  from preprocess import get_X, get_y
  from nn import ANN
  from utils import CustomDataset
  from tqdm.auto import tqdm

  device = torch.device(args.device)

  train_df = pd.read_csv(args.data_train)
  X_trn = get_X(train_df)
  y_trn = get_y(train_df)[:,np.newaxis]

  ds = CustomDataset(X_trn, y_trn)
  dl = DataLoader(ds, batch_size=args.batch_size, shuffle=args.shuffle)

  model = ANN(X_trn.shape[-1], args.hidden_dim).to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

  pbar = range(args.epochs)
  if args.pbar:
    pbar = tqdm(pbar)
  for _ in pbar:
    loss = train_one_epoch(model, nn.functional.binary_cross_entropy, optimizer, dl, device)
    pbar.set_postfix(trn_loss=loss)

  torch.save(model.state_dict(), args.output)


def get_args_parser(add_help=True):
  import argparse

  parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)

  parser.add_argument("--data-train", default="./data/train.csv", type=str, help="train dataset path")
  # parser.add_argument("--data-test", default="./data/test.csv", type=str, help="test dataset path")
  parser.add_argument("--hidden-dim", default=128, type=int, help="dimension of hidden layer")
  parser.add_argument("--device", default="cpu", type=str, help="device (Use cpu/cuda/mps)")
  parser.add_argument("-b", "--batch-size", default=32, type=int, help="batch size")
  parser.add_argument("--shuffle", default=True, type=bool, help="shuffle")
  parser.add_argument("--epochs", default=300, type=int, metavar="N", help="number of total epochs to run")
  parser.add_argument("--lr", default=0.0001, type=float, help="learning rate")
  parser.add_argument("--pbar", default=True, type=bool, help="progress bar")
  parser.add_argument("-o", "--output", default="./model.pth", type=str, help="path to save output model")
  
  return parser

if __name__ == "__main__":
  args = get_args_parser().parse_args()
  main(args)