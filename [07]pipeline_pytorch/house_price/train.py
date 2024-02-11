import torch
from torch import nn
from torch.utils.data import DataLoader
import torchmetrics
from typing import Optional

def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_function,
    optimizer: torch.optim.Optimizer,
    device: str,
    metric:torchmetrics.Metric,
) -> float:
    '''train_one_epoch
    Args:
        model: model
        data_loader: data loader
        loss_function: loss function
        optimizer: torch optimizer
        device: device
        # metric: metrics to use
    '''
    model.train() # Sets the module in training mode
    total_loss = 0. # initialize, set to float
    
    for X, y in data_loader: # data_loader = 이미 batch로 불러와서 넣은 상태
        X, y = X.to(device), y.to(device)

        # Compute prediction error (loss)
        pred = model(X) # output
        loss = loss_function(pred, y)   # gradient 접근할때에는 항상 미분을 고려해야 함.  # torch.sqrt(loss)
 
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # metric updates
        metric.update(pred, y)
        # 직접계산: total loss for each batch (loss.item(): every batch)
        # total_loss += loss.item() * len(y) # len(y): 데이터 입력 크기에 따라 다른, loss에 미치는 영향을 조절, 가중치 부여하는 역할: 정규화
        
    # return total_loss/len(data_loader.dataset)# = 각 데이터에 대한 평균 loss 값
    



# train set 통으로 돌리기
def main(cfg):
    """main: run train set as a whole
    cfg: config.py
    """
    import numpy as np
    import pandas as pd
    from torch.utils.data.dataset import TensorDataset
    from nn import ANN
    from tqdm.auto import trange

    train_params = cfg.get("train_params")
    device = torch.device(train_params.get("device"))

    # get preprocessed X, y csv files
    files = cfg.get("files")
    X = torch.tensor(pd.read_csv(files.get('X_csv'), index_col=0).to_numpy(dtype=np.float32))
    y = torch.tensor(pd.read_csv(files.get('y_csv'), index_col=0).to_numpy(dtype=np.float32))

    # Tensor Dataset load
    # train
    dl_params = train_params.get("data_loader_params")
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, **dl_params)

    # Model
    Model = cfg.get("model")
    model_params = cfg.get("model_params")
    model_params["input_dim"] = X.shape[-1] # 여기에 자동으로 input_dim 넣어준다
    model = Model(**model_params).to(device)
    # print(model) # To check the model used here
    
    Optim = train_params.get("optim")
    optim_params = train_params.get("optim_params")
    optimizer = Optim(model.parameters(), **optim_params)

    # Evaluation Scores - kfold 
    # scores = mse()
    
    loss = train_params.get("loss") # Loss function 
    metric = train_params.get("metric")
    values = [] # for metrics

    # training and progress bar display
    pbar = trange(train_params.get("epochs"))  #tqdm bar
    for _ in pbar:
        train_one_epoch(model, dl, loss, optimizer, device, metric)
        values.append(metric.compute().item())
        metric.reset()
        pbar.set_postfix(trn_loss=values[-1]) # last metric 
    # Save pre-trained weight 
    torch.save(model.state_dict(), files.get("output"))

### parser
def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="Pytorch Model Trainer", add_help=add_help)
    parser.add_argument("-c", "--config", default="./config.py", type=str, help="configuration file")

    return parser

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    exec(open(args.config).read())
    main(config)