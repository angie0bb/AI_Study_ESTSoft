# Pytorch 튜토리얼 (3) - A to Z 파이프라인 짜보기

태그: HousePrice, Titanic, 분류, 회귀
No.: 7

<aside>
💡 파일을 쪼개고, 빠르고 정확하게 결과를 비교할 수 있도록 설계하자

</aside>

### Process

1) 데이터 전처리

- 데이터 로드 - pd.DataFrame
- 결측값, 이상치 처리
- EDA를 통해 필요한 features를 결정
- 벡터화 - numpy

2) 모델 Train

- (K-fold Train & 결과비교)
- 통으로 Train & 결과 확인
- 결과 및 파라미터 저장

3) 모델 Test

- 최종 선택 모델 Test set
- 모델 평가
- 예측 결과 Submission

## (1) Pipeline A to Z - Classification with Titanic Dataset

- data source: [https://www.kaggle.com/competitions/titanic](https://www.kaggle.com/competitions/titanic)
- Random Forest
- Simple NN model

```
### Process
 - X, y 분리된 데이터 준비
 - 데이터를 Pytorch Dataset으로 변환 
 - Dataloader를 이용해서 batch size별로 가져오기
 - Hyperparameters 설정 (optimizer, lr 등)
 - 모델 학습 (model, dataloader, loss function, optimizer)
    - train set 학습
    - 평가
    - 학습한 best weights 저장 -> 요거 따로 해보기!
 - test set 검증
```

- 디렉토리 구조
    - /data
    - metric.py
    - [nn.py](http://nn.py): models
    - preprocess.py
    - rf.py: random forest
    - train.py: training function
    - utils.py

## (2) Pipeline A to Z - Regression with House Price Dataset

- data source: [https://www.kaggle.com/c/home-data-for-ml-course](https://www.kaggle.com/c/home-data-for-ml-course)
- config와 parser를 이용해서 빠른 실험을 할 수 있는 파이프라인을 구축해본다.
- 디렉토리 구조
    - /data
    - [config.py](http://config.py)
    - [kfold.py](http://kfold.py)
    - nn.py
    - test.py
    - train.py
    - utils.py
    - [validate.py](http://validate.py)
    

## (3) Scratch부터 파이프라인 짜보기

### 참고 자료

- [Pytorch Quickstart](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)

### 배운 점

- 빠르게 학습이 되는 지, 실험을 먼저 해보고 싶다면 quickstart처럼 최대한 가볍게 학습/평가를 진행해보자
- Scratch부터 시작하는 연습하기
    - 해보니: 코드 작성 전에 구조를 빠삭하게 알고 있어야 바로바로 나올 것 같음.
        - 기존 파이프라인은 cv도 하고, trian따로, test따로 이런식으로 진행했는데, 노트북에서 진행할 수 있는 가벼운 버전의 파이프라인도 연습해보자.
- 질문: 강사님이 이번에 노트북에서 작성하신 코드를 보니, eval()쪽에서 따로 dataloader 루프를 하지 않고, 밖에서 next()를 통해서 불러옴. 이해한 내용 맞는지 질문하자.
    - input, target = next(iter(tst_dl))  ### test함수에서 dataloader 따로 가져오는 loop부분이 없었음. 여기에서 처리하는 것 같다.
        # iter()는 데이터 로더를 iterable 객체로 만들어줌. 이때, test셋이기 때문에 따로 batch로 쪼개서 가져오는 게 아니라, 
        # 데이터셋 전체를 통으로 가져옴 (그래야 전체 테스트 데이터셋에서 모델이 얼마나 잘 수행되는지를 빠르게 평가할 수 있음)
        # next()는 iterable 객체에서 다음 요소를 가져오는 것. 여기에서는 통으로 가져온 데이터를 한 번에 모델에 전달해서 평가 수행

### 구조

```
# ------------------ 데이터 로드 --------------- #
# Dataset 불러오기 (train, test)
# 원하는 Batch 사이드 별로 불러오기 위해 Dataloader에 넣기
# ------------------ 모델 설계 ----------------- #
# nn 모델 정의 (init, forward)
# model.to(device)
# ------------------ 모델 학습, 최적화 ----------- #
# loss function
# optimizer
# train 함수 정의(dataloader, model, loss, optimizer)
	# batch size별로 train loop 돌기
		# compute prediction error
		# backpropagation
# test 셋에서도 학습 잘 되는지 확인하기(dataloader, model, loss)
	# batch size를 테스트셋 통으로 설정해서 돌리기
		# with torch.inference (optimize 하지 않고 확인: no gradient)
			# loss or metric 계산
# 학습 loop 돌기 (epoch 설정: epoch마다 모델은 파라미터를 학습하고 업데이트함)
	# 이때 train set, test set 각각 함수 사용해서 동시에 확인할 수 있음
# 원하는 모델을 torch.save
# 나중에 해당 모델을 load해서 prediction을 얻을 수 있음.
```

### Task 1 다시 해보기

- 메인 부분만

```python
# train 함수 for one epoch
def train(dataloader, model, loss_function, optimizer):
    model.train() # train 모드 진입
    total_loss = 0. # initialize 
    for input,target in dataloader: # dataloader에서 알아서 batch별로 꺼내올거임
        input, target = input.to(device), target.to(device)
        # compute prediction error
        pred = model(input)
        loss = loss_function(pred, target)
        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # loss 직접 계산
        total_loss += loss.item()*len(pred)  # loss.item() loss for every batch 
    return total_loss/len(dataloader.dataset) # 평균 loss값

# test 셋에 넣어서 바로바로 metric 확인해볼 용도, test셋에서도 학습 잘 되는지 확인부터 해야하니까
def test(model, input, target, loss_function, metric, return_pred:bool=False):  # return_pred이 True일때만, prediction값 return
    model.eval() # eval 모드 진입
    with torch.inference_mode():   ################ 
        # TODO 질문: # https://colab.research.google.com/drive/1uO3Ep1GNB0GMEuXnzy1eqvX9ezovHFnm#scrollTo=VgDZYq-uFhy-
        # compute prediction error
        pred = model(input.to(device))
        tst_loss = loss_function(pred, target.to(device)).item()
        # metric도 넣어서 평가하자
        tst_mae = metric(pred, target.to(device)).item()
        # df_metric = pd.DataFrame([loss, mae], index = ["tst_loss: MSE", "tst_metric: MAE"])
        pred_np = pred.numpy()  ### prediction 값 내보내기 위함
    if return_pred:
        return tst_loss, tst_mae, pred_np
    return tst_loss, tst_mae
```

```python
from tqdm.auto import trange # for progress bar visualization

model = ANN(input_dim = window_size, output_dim= prediction_size, hidden_dim=128, activation=F.relu)
model.to(device)

loss_function = F.mse_loss
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
metric = F.l1_loss # mae

epochs = 2000
pbar = trange(epochs)
for i in pbar:
    trn_mse = train(dataloader=trn_dl, model=model, loss_function=loss_function, optimizer=optimizer)
    input, target = next(iter(tst_dl))  ### test함수에서 dataloader 따로 가져오는 loop부분이 없었음. 여기에서 처리하는 것 같다.
    # iter()는 데이터 로더를 iterable 객체로 만들어줌. 이때, test셋이기 때문에 따로 batch로 쪼개서 가져오는 게 아니라, 
    # 데이터셋 전체를 통으로 가져옴 (그래야 전체 테스트 데이터셋에서 모델이 얼마나 잘 수행되는지를 빠르게 평가할 수 있음)
    # next()는 iterable 객체에서 다음 요소를 가져오는 것. 여기에서는 통으로 가져온 데이터를 한 번에 모델에 전달해서 평가 수행
    tst_loss, tst_mae = test(model = model, input=input, target=target, loss_function = loss_function, metric = metric)
    
    pbar.set_postfix({"trn_mse":trn_mse, "tst_mse": tst_loss, "tst_mae": tst_mae})
```