import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchmetrics
from dataclasses import dataclass, field
from typing import Type, Optional
import pandas as pd

def validate_one_epoch( 
    model: nn.Module,
    data_loader: DataLoader,
    loss_function,
    device: str,
    metric:torchmetrics.Metric,
    ) -> float:
    '''train_one_epoch
    Args:
        model: model
        data_loader: data loader
        loss_function: loss function
        device: device
        # metric: metrics to use
    '''
    model.eval()  # turn model into evaluation mode
    total_loss = 0.

    with torch.inference_mode(): # Same with torch.no_grad -> (legacy) 원래 no_grad는 gradient도 안 하고, 가중치 업데이트 안함. (대신 속도 빠름)
        # torch.no_grad = 가중치 업데이트 안함, inference_mode = 가중치 업데이트 함
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            
            # Compute prediction error (loss)
            pred = model(X)
            metric.update(pred, y)
            # total_loss += loss_function(pred, y).item() * len(y)
            # ####### 강사님 코드에서는 evaluate()에서 total_loss 계산 안 함
            # if metric is not None:
            #     metric.update(pred, y)
        
    # return total_loss/len(data_loader.dataset) 
                
