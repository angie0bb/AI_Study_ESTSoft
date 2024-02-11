import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchmetrics

from sklearn.model_selection import StratifiedKFold
from nn import ANN
from torch.utils.data import TensorDataset
from tqdm.auto import tqdm
from tqdm.auto import trange
from train import train_one_epoch
from validate import validate_one_epoch
import pandas as pd
import numpy as np
# apply parameters  

def kfold_validate(cfg): # k_fold, train_one_epoch, evaluate

    train_params = cfg.get("train_params")
    device = torch.device(train_params.get("device"))

    n_splits = 5
    k_fold = StratifiedKFold(n_splits = n_splits, shuffle = False)  ######shuffle = True   
    # True로 하면 모델이 좀 더 일반화 되겠지...섞이니까...
    
    # get preprocessed train_X, y csv files
    files = cfg.get("files")
    train_X = pd.read_csv(files.get('X_csv'), index_col=0).to_numpy(dtype=np.float32)
    train_y = pd.read_csv(files.get('y_csv'), index_col=0).to_numpy(dtype=np.float32)
    X, y = torch.tensor(train_X), torch.tensor(train_y)

    # Apply ANN to 5 different folds 
    Model = cfg.get("model")
    model_params = cfg.get("model_params")
    model_params["input_dim"] = X.shape[-1] # 여기에 자동으로 input_dim 넣어준다
    nets = [Model(**model_params).to(device) for i in range(n_splits)] 
    #   loss_results = {}

    for i, (train_idx, val_idx) in enumerate(k_fold.split(X, y)): # split(): split to train and valid set
      
        # train set for each fold
        X_trn, y_trn = torch.tensor(X[train_idx]), torch.tensor(y[train_idx])
        
        # validation set for each fold
        X_val, y_val = torch.tensor(X[val_idx]), torch.tensor(y[val_idx])

        # Tensor Dataset load
        # train
        dl_params = train_params.get("data_loader_params")
        ds = TensorDataset(X_trn, y_trn)
        ds_val = TensorDataset(X_val, y_val)

        dl = DataLoader(ds, **dl_params)
        dl_val = DataLoader(ds_val, batch_size = len(ds_val), shuffle=False) # validation set은 batch 통으로 돌리기 

        net = nets[i]
        # Choose optimizer
        Optim = train_params.get("optim")
        optim_params = train_params.get("optim_params")
        optimizer = Optim(net.parameters(), **optim_params)

        loss = train_params.get("loss") # Loss function 
        # metrics
        metric = train_params.get("metric")
        metrics = {'trn_rmse': [], 'val_rmse': []} # for metrics

        # loss, accuracy check with progress bars
        pbar = trange(train_params.get("epochs")) # epoch = 500 for each fold
        for _ in pbar: # display metrics on a pbar
            # accuracy = BinaryAccuracy().to(device)
            # TODO value 저장할 리스트?
            train_one_epoch(net, dl, loss, optimizer, device, metric)  # loss 로 RMSE를 쓰면 local minima에 빠지기 쉬움
            trn_rmse = metric.compute().item()
            metric.reset()
            validate_one_epoch(net, dl_val, loss, device, metric)
            val_rmse = metric.compute().item()
            metric.reset()
            pbar.set_postfix(trn_rmse = trn_rmse, val_rmse = val_rmse)

        metrics['trn_rmse'].append(trn_rmse)
        metrics['val_rmse'].append(val_rmse)
    return pd.DataFrame(metrics)
        
def get_args_parser(add_help=True):
  import argparse
  
  parser = argparse.ArgumentParser(description="Pytorch K-fold Cross Validation", add_help=add_help)
  parser.add_argument("-c", "--config", default="./config.py", type=str, help="configuration file")

  return parser

if __name__ == "__main__":
  args = get_args_parser().parse_args()
  exec(open(args.config).read())
  kfold_validate(config)

  # results = kfold_validate(config)
  # print(results)