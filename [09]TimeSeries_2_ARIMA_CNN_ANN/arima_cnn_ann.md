# 시계열 (2) - ARIMA, CNN, ANN 성능 비교

태그: ANN, ARIMA, CNN, 시계열
No.: 9

## ARIMA, CNN, ANN 성능 비교

### ARIMA

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-darkgrid')

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm

import torch
import torch.nn as nn
import torch.nn.functional as F

print(sm.datasets.sunspots.NOTE)
data = sm.datasets.sunspots.load_pandas().data

data.index = pd.Index(sm.tsa.datetools.dates_from_range("1700", "2008"))
data.index.freq = data.index.inferred_freq
del data["YEAR"]

trn, tst = data[:-20], data[-20:]

# plot
"""
ax = trn.plot(title="Sunspots", label='trn', figsize=(12,5))
tst.plot(label='tst', ax=ax)
plt.legend()
plt.show()
"""

from statsmodels.tsa.arima.model import ARIMA

trn, tst = data.SUNACTIVITY[:-20], data.SUNACTIVITY[-20:]
p, d, q = 9, 0, 0 # AR param = 9, I param = 0, MA param = 0
arma_mod80 = ARIMA(trn, order=(p, d, q)).fit() #ARMA(9, 0)

arima_prd = pd.DataFrame(arma_mod80.predict("1989", "2008")) # dynamic=True : 금요일 방식의 task_2 방식, dynamic=False: task_1 방식

arima_prd.rename(columns={"predicted_mean": "ARIMA"}, inplace=True)
```

### ANN

```python
# dataset
from torch.utils.data import Dataset, DataLoader
class TimeseriesDataset(Dataset):
  def __init__(self, data, window_size=10, prediction_size=5):
    self.data = data
    self.window_size = window_size
    self.prediction_size = prediction_size

  def __len__(self):
    return len(self.data) - self.window_size - self.prediction_size

  def __getitem__(self, idx):
    inp = self.data[idx:(idx + self.window_size)]
    tgt = self.data[(idx + self.window_size):(idx + self.window_size + self.prediction_size)]
    return inp, tgt

window_size=8
prediction_size=1
test_length=20
trn_ds = TimeseriesDataset(
  data.to_numpy(dtype=np.float32),
  window_size,
  prediction_size
)
trn_dl = DataLoader(trn_ds, shuffle=True, batch_size=32)

tst_ds = TimeseriesDataset(
  data.to_numpy(dtype=np.float32)[-test_length-window_size:],
  window_size,
  prediction_size
)
tst_dl = DataLoader(tst_ds, shuffle=False, batch_size=len(tst_ds))

# model
import torch.nn as nn

activation_list = {"sigmoid": nn.Sigmoid(), "relu": nn.ReLU(), "tanh": nn.Tanh(), "prelu": nn.PReLU()}
class ANN(nn.Module):
  def __init__(self, input_dim: int=5, hidden_dim: list=[64, 32], activation:str="sigmoid", use_dropout: bool=False, drop_ratio: float=0.5, output_dim:int =5):
    """ Artificial Neural Network(ANN) with Linear layers
      the model structure is like below:
      
      Linear
      Dropout
      Activation
      
      Linear
      Dropout
      ...
    
    
    Args:
      input_dim (int): dimension of input
      hidden_dim (list): list of hidden dimension. the length of 'hidden_dim' means the depth of ANN
      activation (str): activation name. choose one of [sigmoid, relu, tanh, prelu]
      use_dropout (bool): whether use dropout or not
      drop_ratio (float): ratio of dropout
    """
    super().__init__()

    dims = [input_dim] + hidden_dim # [5, 128, 128, 64, 32]
    
    self.dropout = nn.Dropout(drop_ratio)
    self.identity = nn.Identity()
    self.activation = activation_list[activation]
    self.relu = activation_list["relu"]
    
    model = [[nn.Linear(dims[i], dims[i+1]), self.dropout if use_dropout else self.identity, self.activation] for i in range(len(dims) - 1)]

    output_layer = [nn.Linear(dims[-1], output_dim)] # Delete sigmoid for regression model
    
    self.module_list= nn.ModuleList(sum(model, []) + output_layer)
  
  def forward(self, x):
    x = x.squeeze()
    for layer in self.module_list:
         x = layer(x)
    return x

# train_one_epoch
import torchmetrics

def train_one_eph(
    model: nn.Module,
    criterion: callable,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    metric: torchmetrics.Metric,
    device: str,
) -> None:
    """train one epoch

    Args:
        model: model
        criterion: loss
        optimizer: optimizer
        data_loader: data loader
        device: device
    """
    model.train()
    for X, y in data_loader:
        X, y = X.to(torch.float32), y.to(torch.float32)
        X = X.squeeze()
        y = y.reshape(y.shape[0], 1)
        # print(X.shape, y.shape)
        X, y = X.to(device), y.to(device)
        output = model(X)
        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        metric.update(output, y)
    return metric.compute().item()

# train
from tqdm.auto import trange

ann_model = ANN(input_dim=8, output_dim=1, activation="relu")
ann_model.train()

optimizer = torch.optim.AdamW(ann_model.parameters(), lr=0.0001)
device="cpu"
criterion=nn.MSELoss()
metric = torchmetrics.MeanAbsoluteError().to(device)

pbar = trange(1000)
for _ in pbar:
  loss = train_one_eph(ann_model, criterion, optimizer, trn_dl, metric, device)
  metric.reset()
  pbar.set_postfix({'loss': loss})

#eval
# 단기예측
ann_preds_short = torch.zeros(1)
with torch.inference_mode():
  for X, y in tst_dl:
    X, y = X.to(torch.float32), y.to(torch.float32)
    # print(y)
    X, y = X.to(device), y.to(device)
    output = ann_model(X)
    # print(result)
    if not torch.any(ann_preds_short):
      ann_preds_short = output[:, 0]
      # print(result)
    else:
      ann_preds_short = torch.cat((ann_preds_short, output[:,0]), 0)

# 장기예측
window_size = 8
prediction_size = 1
test_size=20
preds = []
# print(type(x), type(y))
x, y = trn_ds[len(trn_ds)]
for _ in range(test_length):
  # print(type(x), type(y))
  x = np.concatenate([x,y])[-window_size:]
  y = ann_model(torch.tensor(x, dtype=torch.float32)) # your model
  y = y.detach().numpy().reshape(1,prediction_size)
  preds.append(y)

preds = np.concatenate(preds).squeeze()
print(preds.shape)

# 단기예측 plot
tst = data.SUNACTIVITY[-20:-1]
ax = tst.plot(label="TRUE")
pd.DataFrame({"ANN_short": ann_preds_short}, index=tst.index).plot(ax=ax)
plt.show()

# 장기예측 plot
trn, tst = data.SUNACTIVITY[:-20], data.SUNACTIVITY[-20:]
ax = tst.plot(label="TRUE")
pd.DataFrame({"ANN_long": ann_preds_long}, index=tst.index).plot(ax=ax)
```

### CNN

```python
trn_t = torch.tensor(trn.SUNACTIVITY.to_numpy(dtype=np.float32)[np.newaxis, np.newaxis])
tst_t = torch.tensor(tst.SUNACTIVITY.to_numpy(dtype=np.float32)[np.newaxis, np.newaxis])

class ARMA(nn.Module):
  def __init__(self, p:int, q:int):
    assert p>=0
    assert q>=0
    assert p or q

    self.p, self.q = p, q
    super().__init__()

    if p:
      self.ar_conv = nn.Conv1d(1, 1, p, bias=False)   # (N, 1, L) -> (N, 1, L - (p-1))
    if q:
      self.ma_conv = nn.Conv1d(1, 1, q, bias=False)   # (N, 1, L) -> (N, 1, L - (q-1))

  def forward(self, y):
    p_shift, q_shift = 0, 0

    res = y - y.mean()
    if self.p:
      p_shift = self.p-1
      # (N, 1, L)
      ar = self.ar_conv(y)
      # (N, 1, L - (p-1))
      res = res[:,:,p_shift:] - ar

    if self.q:
      q_shift = self.q-1
      # (N, 1, L - (p-1))
      ma = self.ma_conv(res)
      # (N, 1, L - (p-1) - (q-1))
    else:
      ma = 0.
    return y.mean() + ar[:,:,q_shift:] + ma

  def predict(self, y, steps=1):
    y = y.clone().reshape(1,1,-1)
    # print(y.shape)
    for i in range(steps):
      prd = self.forward(y)[:,:,-1:]
      y = torch.concat([y,prd], axis=-1)
      # print(y.shape)
    return y[:,:,-steps:].flatten()

from tqdm.auto import trange

arma = ARMA(8,1)
optim = torch.optim.AdamW(arma.parameters(), lr=0.0001)

pbar = trange(20000)
for _ in pbar:
  arma.train()
  prd = arma(trn_t[:,:,:-1])
  optim.zero_grad()
  loss = F.mse_loss(prd, trn_t[:,:,8:])
  loss.backward()
  optim.step()
  pbar.set_postfix({'loss': loss.item()})

arma.eval()
with torch.inference_mode():
  arma_cnn_prd = arma.predict(trn_t, steps=60)

# plot
'''
ax = tst.plot(label="TRUE")
pd.DataFrame({"CNN": arma_cnn_prd}, index=tst.index).plot(ax=ax)
'''
```

### Metrics

- MAPE: linear scale에는 상관없이 동일한 값이 나옴 (log,,이런 scaling은 안 됨)
- MAE: inverse scale 해줘야 함

### 비교

1) ARIMA

![Untitled](%E1%84%89%E1%85%B5%E1%84%80%E1%85%A8%E1%84%8B%E1%85%A7%E1%86%AF%20(2)%20-%20ARIMA,%20CNN,%20ANN%20%E1%84%89%E1%85%A5%E1%86%BC%E1%84%82%E1%85%B3%E1%86%BC%20%E1%84%87%E1%85%B5%E1%84%80%E1%85%AD%2086067f30090a4dca9eeb7701acface85/Untitled.png)

2) SARIMA

![Untitled](%E1%84%89%E1%85%B5%E1%84%80%E1%85%A8%E1%84%8B%E1%85%A7%E1%86%AF%20(2)%20-%20ARIMA,%20CNN,%20ANN%20%E1%84%89%E1%85%A5%E1%86%BC%E1%84%82%E1%85%B3%E1%86%BC%20%E1%84%87%E1%85%B5%E1%84%80%E1%85%AD%2086067f30090a4dca9eeb7701acface85/Untitled%201.png)

3) ANN

![Untitled](%E1%84%89%E1%85%B5%E1%84%80%E1%85%A8%E1%84%8B%E1%85%A7%E1%86%AF%20(2)%20-%20ARIMA,%20CNN,%20ANN%20%E1%84%89%E1%85%A5%E1%86%BC%E1%84%82%E1%85%B3%E1%86%BC%20%E1%84%87%E1%85%B5%E1%84%80%E1%85%AD%2086067f30090a4dca9eeb7701acface85/Untitled%202.png)

4) CNN 

![Untitled](%E1%84%89%E1%85%B5%E1%84%80%E1%85%A8%E1%84%8B%E1%85%A7%E1%86%AF%20(2)%20-%20ARIMA,%20CNN,%20ANN%20%E1%84%89%E1%85%A5%E1%86%BC%E1%84%82%E1%85%B3%E1%86%BC%20%E1%84%87%E1%85%B5%E1%84%80%E1%85%AD%2086067f30090a4dca9eeb7701acface85/Untitled%203.png)