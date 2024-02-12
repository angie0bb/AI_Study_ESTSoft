# Pytorch 튜토리얼 (2) - 타이타닉 생존자 분류

태그: ANN, Scikit-learn, pandas
No.: 6

## Pandas Tutorial

- pandas 기능 써보기
    - source: [데잇걸즈_10분판다스](https://www.notion.so/Pytorch-2-1c02b3645c974a1fbe8f44a8fe20f761?pvs=21)

## EDA with Titanic Dataset

### EDA 할 때 주의할 점

- 집계함수를 사용해서 새로운 column을 만들 수 있다.
- 상관관계 확인하기
- 값의 가중편향: 변수끼리에도 값의 편차가 다른 경우 정규화가 필요함
    - 데이터의 분포를 보고 갑자기 큰 값이 있는지, 고려해서 정규화를 해야 함
        - Sigmoid함수를 적용했을 때를 생각해보자.
- 결측값 비율 확인하기 (/len())
- FARE→ ticket의 총요금이라면? 사람 수로 나누어서 해야함
- Sibling, Spouse → 있는지 없는지가 중요하지, 사실 수는 중요하지 않음
    - 0,1로 나누어서 상관관계 분석을 하는 것과, 수치가 있는 상태에서와는 차이가 큼
    - 동승자 여부로 나누어서 판별하면 좋았을 것 같음.
- 데이터를 살펴볼때에는 창의력을 발휘하자!
    - 버릴때에는 살릴 수 없던 분명한 이유를 꼭 알려줘야 함.

### 데이터 정리와 모델의 관계

- 내가 정리한 데이터가 모델이 쉽게 인식할 수 있어야 함.
- 0과 1만 있는 값 → Relu, Sigmoid등 층이 몇개 생기는지 여부가 달라짐

## Random Forest

- 결정 트리를 랜덤하게 만들고, 각 트리들의 예측을 사용해 최종 예측을 만든다.
    - 랜덤하게 샘플과 특성을 선택하기 때문에, overfitting을 방지하고 안정적인 성능을 기대할 수 있음.
- 각 트리를 훈련하기 위한 데이터를 train data에서 샘플을 추출해 랜덤하게 만든다. 이때 샘플은 중복될 수도 있음. (bootstrap sample → 뽑은 샘플을 다시 넣고 뽑기 때문에 중복이 생길 수 있음, replacement)
- 각 노드를 분할할 때 전체 특성 중에서 일부 특성을 무작위로 고르고, 이 중에서 최선의 분할을 찾음.
    - `RandomForestClassifier`: 기본적으로 전체 특성 개수의 제곱근($\sqrt{features}$)만큼의 특성을 선택. 특성 개수가 총 16개면 4개씩 랜덤 선택. 이런 방식으로 n개의 결정 트리들을 훈련한 뒤, 각 트리의 클래스별 확률을 평균해서 가장 높은 확률을 가진 클래스를 예측값으로 제공
    - `RandomForestRegressor`: 전체 특성 사용, 각 트리의 예측을 평균

```python
from sklearn.ensemble import RandomForestClassifier
X = [[0, 0], [1, 1]] # X.shape = (n_samples, n_features) # to_numpy() 활용 가능
Y = [0, 1] # Y.shpae = (n_samples, )
clf = RandomForestClassifier(n_estimators=100) # n_estimators: number of trees
# max_depth: if default = None, nodes are expanded until all leaves are pure or 
# untill all leaves contain less than min_samples_split samples
# n_jobs: number of jobs to run in parallel, -1 = using all cpu processors
clf = clf.fit(X, Y)
```

## Cross Validation

- To begin with: Train & Test Set - Validation Set
    - 테스트 세트를 최대한 아끼기 위해, 트레인 세트를 validation set, train set로 다시 나누기
        
        ```python
        # Validation Set split from train set 
        from sklearn.model_selection import train_test_split
        train_input, valid_input, train_target, valid_target = 
        		train_test_split(data, target, test_size = 0.2, random_state = 1)
        # Allowed inputs are lists, numpy arrays, scipy-sparse matrices or pandas dataframes.
        # test_size = validation set을 몇 프로로 할건지
        # random_state = 섞는 방식 고정 
        ```
        
- K-fold cross validation
    - validation set을 나누어서 테스트하는 검증 방법을 k번 반복해서 검증 점수의 평균값을 활용
        
        ```python
        # 5-fold cross validation
        """cross_validate 함수
        - train set을 섞어주진 않는다. train_test_split()함수를 사용했으면 train set을 
        섞은 상태에서 진행하기 때문에 상관 없지만, 그렇지 않다면 splitter를 따로 지정해주어야 함.
        
        splitter = StratifiedKfold(n_splits = 10, shuffle = True, random_state = 1)
        scores = cross_validate(dt, train_input, train_target, cv = splitter)
        """
        from sklearn.model_selection import cross_validate
        scores = cross_validate(dt, train_input, train_target) # default = 5-fold
        np.means(scores['test_score'] # 각 폴드별 점수의 평균
        
        # 분류 모델이면, 타깃 클래스를 골고루 나누기 위해 StratifiedKFold 사용
        scores = cross_validate(dt, train_input, train_target, cv = StratifiedKFold())
        ```