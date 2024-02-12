# 시계열 (3) - RNN, LSTM

태그: LSTM, RNN, 시계열
No.: 10

## RNN

- NN이 현재 값만으로 다음 값을 예측할 순 없음 
→ 지금까지 나왔던 이전 값들에 대한 추가정보(hidden state)로써 활용함으로써 다음 값을 예측
- 현재 값(이전 step의 예측 값)과 이전 step의 hidden state를 통합하여 다음 예측값 및 현재에 대한 hidden state 도출
- Hidden state
    - $W_{hy}$: hidden state를 linear map을 통해 input dimension과 동일하게 변환
    - $W_{hh}$: hidden state를 linear map을 통해 hidden dimension과 동일하게 변환, hidden state를 업데이트 해주는 방법 포함
        - 결국 두 가지 모두 동일한 hidden state로부터 기원(동일한 vector를 서로 다른 linear map으로 매핑)
        - 모든 step에 대해 $W_{hy}, W_{hh}$ 는 동일한 layer
- output: 2개 => 1) $\hat{y}_t$: 다음에 대한 예측 2) $W_{hh}$: 현재의 hidden state

## LSTM

![Untitled](%E1%84%89%E1%85%B5%E1%84%80%E1%85%A8%E1%84%8B%E1%85%A7%E1%86%AF%20(3)%20-%20RNN,%20LSTM%204357107a4991455c8ff0128031c77162/Untitled.png)

- 2개의 hidden state: 1) hidden state 2) memory(cell state)
- output과 hidden state가 정확히 동일
- LSTM of torch: [torch.nn.LSTM](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
    - `torch.nn.LSTM(self, input_size, hidden_size, num_layers=1, batch_first=True, proj_size=0)`
        - `x`: output (aka hidden states), shape: `(batch_size, sequence_length, hidden_size)`
        - `hn`: the final hidden state, shape: `(num_layers, batch_size, hidden_size)`
        - `cn`: the final cell state, shape: `(num_layers, batch_size, hidden_size)`
    

## Dataset: Sunspot Activity

```python
print(sm.datasets.sunspots.NOTE)
data = sm.datasets.sunspots.load_pandas().data

data.index = pd.Index(sm.tsa.datetools.dates_from_range("1700", "2008"))
data.index.freq = data.index.inferred_freq
del data["YEAR"]

tst_size = 20

data_mw = data.copy()
data_mw['rolling_avg'] = data.SUNACTIVITY.rolling(12).mean()
data_mw = data_mw.dropna()

trn, tst = data_mw[:-tst_size], data_mw[-tst_size:]

scaler = MinMaxScaler()
scaler_ra = MinMaxScaler()

trn_scaled, tst_scaled = trn.copy(), tst.copy()

trn_scaled['SUNACTIVITY'] = scaler.fit_transform(trn.SUNACTIVITY.to_numpy(np.float32).reshape(-1,1))
trn_scaled['rolling_avg'] = scaler_ra.fit_transform(trn.rolling_avg.to_numpy(np.float32).reshape(-1,1))

tst_scaled['SUNACTIVITY'] = scaler.transform(tst.SUNACTIVITY.to_numpy(np.float32).reshape(-1,1))
tst_scaled['rolling_avg'] = scaler_ra.transform(tst.rolling_avg.to_numpy(np.float32).reshape(-1,1))

trn_scaled = trn_scaled.to_numpy(np.float32)
tst_scaled = tst_scaled.to_numpy(np.float32)

ax = trn.plot(figsize=(12,5))
tst.plot()
```

## 2 가지 데이터셋 구성

### 1. 시계열 전체를 처음부터 외워서 다음을 맞추기(StatefulLSTM)

- 미니배치 없이 전체 데이터셋 정보를 저장 후, 다음 예측
- 이전 state에 대한 정보가 저장되어있어야 정확한 예측 가능
    - state까지 같이 관리해야한다는 면에서 Transformer 대비 단점

```python
class StatefulLSTM(nn.Module):
  def __init__(self, input_size, hidden_size, output_size, num_layers):
    super().__init__()
    self.reset_state()
    self.rnn = nn.LSTM(input_size, hidden_size, num_layers)
    self.head = nn.Linear(hidden_size, output_size)

  def reset_state(self, state=None):
    self.state = state

  def forward(self, x):
    assert x.dim() == 2   # (sequence_length, input_size)
    if self.state is None:
      x, (hn, cn) = self.rnn(x)   # state will be set to be zeros by default
    else:
      x, (hn, cn) = self.rnn(x, self.state)   # pass the saved state
    # x.shape == (sequence_length, hidden_size)
    self.reset_state((hn.detach(), cn.detach()))   # save the state
    x = self.head(x)  # (sequence_length, hidden_size) -> (sequence_length, output_size)
    return F.sigmoid(x)

  def predict(self, x0, steps, state=None):
    if state is not None:
      self.reset_state(state)
    output = []
    x = x0.reshape(1,-1)
    for i in range(steps): # 한 칸씩 옆으로 이동
      x = self.forward(x)
      output.append(x)
    return torch.concat(output, 0) #output의 맨 마지막 녀석들만 따오도록

# Dataset
batch_size = 64
trn_x = torch.tensor(trn_scaled[:-1]).split(batch_size)
trn_y = torch.tensor(trn_scaled[1:]).split(batch_size)

tst_y = torch.tensor(tst_scaled)

trn_x[0].shape, trn_y[0].shape

# Model initialize
rnn = StatefulLSTM(2, 8, 2, 1)
rnn.to(device)

optim = torch.optim.AdamW(rnn.parameters(), lr=0.0005)

pbar = trange(1000)
for e in pbar:
  rnn.train()
  rnn.reset_state() # 각 에폭마다 state를 reset해주어야 함. 
  trn_loss = .0
  for x, y in zip(trn_x, trn_y):
    x, y = x.to(device), y.to(device)
    optim.zero_grad()
    p = rnn(x)
    loss = F.mse_loss(p, y)
    loss.backward()
    optim.step()
    trn_loss += loss.item()
  trn_loss /= len(trn)-1

  
  rnn.eval()
  with torch.inference_mode():
    p = rnn.predict(y[-30:].to(device), len(tst_y)) # 마지막 1개를 넣어서 뒤에 20개를 예측
    tst_loss = F.mse_loss(p, tst_y.to(device)).item()
  pbar.set_postfix({'trn_loss': trn_loss, 'tst_loss': tst_loss})

# 만약 중간에 eval이 아닌 모든 학습이 끝난 후 eval을 수행해야 한다면? 
rnn.reset_state()
for x in trn_x:
    x = x.to(device)
    p = rnn(x) # 이래야 rnn 모델 안의 state에 전체 sequence에 대한 정보가 저장됨
```

### 2. **Look-back Window**

- Stateful LSTM에 비해 좋지 않은 결과를 보임
- why?
    - shuffle함으로써 전체 데이터가 아닌 단순히 64개의 연속된 데이터만을 봄
    - > 64개로 묶인 데이터들에 다양한 패턴이 들어가있어야 효과적으로 학습이 가능
- 언제 쓰지?
    - sequence 길이가 말도 안되게 길다면?(ex. 2시간짜리 녹음본, 60Hz 센서 등)
    - > 어쩔 수 없이 잘라서 써야 함
    

```python
# Set Dataset
class TimeSeriesDataset(torch.utils.data.Dataset):
  def __init__(self, ts:np.array, lookback_size:int, shift_size:int):
    self.lookback_size = lookback_size
    self.shift_size = shift_size
    self.data = ts

  def __len__(self):
    return len(self.data) - self.lookback_size - self.shift_size + 1

  def __getitem__(self, i):
    idx = (i+self.lookback_size)
    look_back = self.data[i:idx]
    forecast = self.data[i+self.shift_size:idx+self.shift_size]

    return look_back, forecast

window_size = 64

trn_ds = TimeSeriesDataset(trn_scaled, window_size, 1)
tst_ds = TimeSeriesDataset(np.concatenate([trn_scaled[-window_size:], tst_scaled], axis=0), window_size, 1)

trn_dl = torch.utils.data.DataLoader(trn_ds, batch_size=64, shuffle=True)
tst_dl = torch.utils.data.DataLoader(tst_ds, batch_size=len(tst_ds), shuffle=False)

#Model
class StatelessLSTM(nn.Module):
  def __init__(self, input_size, hidden_size, output_size, num_layers):
    super().__init__()
    self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
    self.head = nn.Linear(hidden_size, output_size)

  def forward(self, x):
    x, _ = self.rnn(x)   # state will be set to be zeros by default
    # x.shape == (batch_size, sequence_length, hidden_size)
    x = self.head(x)  # (batch_size, sequence_length, output_size)
    return F.sigmoid(x)

  def predict(self, x, steps, state=None):
    output = []
    for i in range(steps):
      x = self.forward(x)
      output.append(x[-1:])
    return torch.concat(output, 0)

# Train, eval
rnn = StatelessLSTM(2, 8, 2, 2)
rnn.to(device)

optim = torch.optim.AdamW(rnn.parameters(), lr=0.001)

pbar = trange(1000)
for e in pbar:
  rnn.train()
  trn_loss = .0
  for x, y in trn_dl:
    x, y = x.to(device), y.to(device)
    optim.zero_grad()
    p = rnn(x)
    loss = F.mse_loss(p, y)
    loss.backward()
    optim.step()
    trn_loss += loss.item()
  trn_loss /= len(trn)-1

  rnn.eval()
  with torch.inference_mode():
    x, y = next(iter(tst_dl))
    p = rnn.predict(x[0].to(device), len(tst_scaled))[:,:1]
    tst_loss = F.mse_loss(p, torch.tensor(tst_scaled[:,:1]).view(-1,1).to(device)).item()
  pbar.set_postfix({'trn_loss': trn_loss, 'tst_loss': tst_loss})
```