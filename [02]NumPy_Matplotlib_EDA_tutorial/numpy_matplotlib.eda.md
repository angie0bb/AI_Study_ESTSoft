# NumPy, Matplotlib, EDA 튜토리얼

태그: EDA, Matplotlib, NumPy
No.: 2

# NumPy

<aside>
📌 공식문서 유저 가이드 [(link)](https://numpy.org/doc/stable/user/index.html)

</aside>

- python에서 벡터 활용에 사용되는 라이브러리
- list와 달리 변수형 설정에 예민하고, broadcasting 등 벡터 연산에 유용한 기능들 제공한다.

## Basic Array Operations

### element-wise operations

- 각 원소별로 연산 수행
- 연산하는 두 벡터의 크기가 다르면 수행되지 않음 (모양이 같아야만 수행)
- +, -, *, /, ** 등
- example
    
    ```bash
    data = np.array([1,2])
    ones = np.ones(2, dtype=int)
    
    print(data + ones)
    print(data * ones)
    print(data / ones)
    print(data ** 2)
    
    a = np.array([1,2,3,4])
    b = np.array([[1,2,3],[3,4,5]])
    print(a.sum())
    print(b.sum())
    print(b.sum(axis=0)) # axis 기준으로 더해짐, 연산 이후 해당 axis는 사라짐
    print(b.sum(axis=0).shape) # 2,3 -> 3
    print(b.max(axis=1))
    print(b.mean())
    ```
    

### Broadcasting

> *The term broadcasting describes how NumPy treats **arrays with different shapes** during arithmetic operations. Subject to certain constraints, the smaller array is “broadcast” across the larger array so that they have compatible shapes. Broadcasting provides a means of vectorizing array operations so that looping occurs in C instead of Python. It does this without making needless copies of data and usually leads to efficient algorithm implementations. There are, however, cases where broadcasting is a bad idea because it leads to inefficient use of memory that slows computation.* [**[source]**](https://numpy.org/doc/stable/user/basics.broadcasting.html)
> 
- example
    
    ```bash
    array_example = np.array([[[1,2,3,4],
                            [4,5,6,7]],
                            [[2,3,4,5],
                            [4,5,6,7]],
                            [[3,4,5,6],
                            [4,5,6,7]]], dtype=np.float32)
    print(array_example.shape)
    print(array_example * 3)
    b = np.array([2,3]).reshape(1,2,1)
    print("\nbroadcasting\n")
    print(array_example * b)
    ```
    

### Additional point

- (4) # literal
- (4,) # tuple

# Matplotlib

- 벡터 시각화를 위해 활용되는 라이브러리
- example
    
    ```bash
    import matplotlib.pyplot as plt
    
    x = np.linspace(0,1,100)
    y = 2 * x + 1
    
    plt.scatter(x,y,s=0.5) # 점을 찍는 함수, s: 점의 크기
    plt.plot(x,y) # 선형 함수
    ```
    

# EDA with Linear Model

## EDA(**Exploratory Data Analysis**)

- 탐색적 데이터 분석
- 무지성 접근(여러 알고리즘, 단순한 모델 등에 넣어서 결과 확인)
- 아래 도구들로 어떤 모델을 사용할지 결정
    - 시각화
    - 군집화
    - 차원축소
    - 가볍고 단순한 모델

## Linear Model

- 1차원 선형 모델: $y = ax + b$
- code example
    
    ```python
    class Linear():
        def __init__(self, a:float = 0., b:float = 0.):
            self.a = a
            self.b = b
        def forward(self, x:np.array):
            return self.a*x + self.b
        __call__ = forward # class 자체를 호출할 수 있음. 함수처럼 사용 가능
    
    x = np.linspace(0,1,100)
    y = 2 * x + 1
    
    lin = Linear(1.5, 1.0)
    lin(x)
    ```
    

## Evaluation

- 전제
- $f$: original function
- $x = \{x_1, x_2, \dots, x_n\}\in\mathbb R^n$
- $y = \{y_1, y_2, \dots, y_n\} = \{f(x_1), f(x_2), \dots, f(x_n)\}$
- $\hat{y} = \{\hat{y}_1, \hat{y}_2, \dots, \hat{y}_n\}$: predicted value

### MSE(Mean Squared Error):

- 각 예측값과 실제 값의 차이의 **제곱**을 평균낸 것

$$
\operatorname{MSE}(\hat{y}, y) = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2
$$

### MAE(Mean Absolute Error)

- 각 예측값과 실제 값의 차이의 **절대값**을 평균낸 것
    
    $$
    \operatorname{MAE}(\hat{y}, y) = \frac{1}{n}\sum_{i=1}^n |y_i - \hat{y}_i|
    $$