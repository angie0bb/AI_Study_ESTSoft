# 시계열 (1) - AR with NN, ARIMA

태그: AR, ARIMA, 시계열
No.: 8

### Time Series Decompose

Time Series Components

- $S_t$: Seasonal component (연말연시 효과, 공휴일 효과도 seasonal에 포함)
- $T_t$: Trend-cycle component (추세, 주기)
- $R_t$: Remainder component (그 외 전부, 잔차)

Seasonality가 원래있던 트렌드에 더해져서 표현될수도 있지만, 곱해져서 표현될수도 있음.

- **log를 취해서** 곱셈을 덧셈으로 바꿔버리면 이 고민 해결!
$y_t = S_t + T_t + R_t$ or $y_t = S_t\times T_t\times R_t$.
But $y_t = S_t\times T_t\times R_t$ is equivalent to $\log y_t = \log S_t + \log T_t + \log R_t.$
- Additive Model
    - (1) 시계열 자료에서 추세(trend)를 뽑아내기 위해서 [중심](https://rfriend.tistory.com/502) [이동 평균(centered moving average)](https://rfriend.tistory.com/502)을 이용합니다.
    - (2) 원 자료에서 추세 분해값을 빼줍니다(detrend). 그러면 계절 요인과 불규칙 요인만 남게 됩니다.
    - (3) 다음에 계절 주기 (seasonal period) 로 detrend 이후 남은 값의 합을 나누어주면 계절 평균(average seasonality)을 구할 수 있습니다. (예: 01월 계절 평균 = (2020-01 + 2021-01 + 2022-01 + 2023-01)/4, 02월 계절 평균 = (2020-02 + 2021-02 + 2022-02 + 2023-02)/4).
    - (4) 원래의 값에서 추세와 계절성 분해값을 빼주면 불규칙 요인(random, irregular factor)이 남게 됩니다.
        - 시계열 분해 후에 추세와 계절성을 제외한 잔차(residual, random/irregular factor) 가 특정 패턴 없이 무작위 분포를 띠고 작은 값이면 추세와 계절성으로 모형화가 잘 되는 것
        - 만약 시계열 분해 후의 잔차에 특정 패턴 (가령, 주기적인 파동을 그린다거나, 분산이 점점 커진다거나 등..) 이 존재한다면 잔차에 대해서만 다른 모형을 추가로 적합해볼 수 있음.
- Multiplicative Model

### Autocorrelation

- 다른 변수간이 아니라, 자기 자신간의 correlation임. 예) 이번 기수의 나와, 저번 기수의 나를 비교
- 한 칸씩 shift
    - 실제 correlation 계산할때 nan값은 무시해주면 됨
- $\operatorname{Corr}(y_t, y_{t-1})$, $\operatorname{Corr}(y_t, y_{t-2})$, and so on...
where
- $y_t = [y_0, y_1, \ldots, y_i, y_{i+1}, \ldots, y_n]$
- $y_{t-1} = [\operatorname{nan}, y_0, y_1, \ldots, y_i, y_{i+1}, \ldots, y_{n-1}]$
- $y_{t-2} = [\operatorname{nan}, \operatorname{nan}, y_0, y_1, \ldots, y_i, y_{i+1}, \ldots, y_{n-2}]$
- and so on...

### White noise

- N(0,1), no autocorrelation (=0)
    - strict하게 정의하진 않음. 둘 중에 하나만 충족해도 white noise라고 생각할 수 있음.
- pattern이나 trend가 보이지 않는, 예측할 수 없는 말 그대로 잡음!

### AutoCorrelation Function (ACF), Partial AutoCorrelation Function (PACF)

- ACF의 절대값이 점점 줄어들고, PACF가 어떤 시점에서 똑 떨어지는 경우면, AR 사용해봐도 좋
- 반대로, PACF의 절대값이 점점 줄어들고 있고, ACF가 어떤 시점에서 똑 떨어지는 (더 이상 벗어나지 않는)경우면, AR이 아닌 MA를 사용하는 것이 좋다

```python
fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(211)
fig = plot_acf(data.values.squeeze(), lags=40, ax=ax1)  # 대략 11년의 주기를 가지고 있다를 acf를 보고 알아낼 수 있음
# 시간이 갈 수록 ac의 영향이 줄어들고 있음.
ax2 = fig.add_subplot(212)
fig = plot_pacf(data.values.squeeze(), lags=40, ax=ax2)  # shift된 값의 기여도
# 9년까지만 벗어나고, 그 이후에는 벗어나지 않음.
# -> 이럴땐 AR활용할만 하고, m =9까지 보면 좋다~ 라는 정설

```

![Untitled](%E1%84%89%E1%85%B5%E1%84%80%E1%85%A8%E1%84%8B%E1%85%A7%E1%86%AF%20(1)%20-%20AR%20with%20NN,%20ARIMA%2005c15281c5594e81b9555e68658f4cc6/Untitled.png)

### Stationary

- white noise는 stationary하지만, stationary한 시계열 데이터가 전부 white noise는 아님.
    - mean, variance are constant over time, no autocorrelation(no trend, cycle)
    - but it doesn't have to be a "random"
- ADF (Augmented Dickey-Fuller Test)
    - if p-value < 0.05, it is stationary (cannot reject) 95% confidence interval
        - 90% confidence interval p-value < 0.1
    
    ```python
    from statsmodels.tsa.stattools import adfuller
    
    adf = adfuller(wn)
    print('ADF Statistic: {}'.format(adf[0]))
    print('p-value: {}'.format(adf[1]))
    print('Critical Values:')
    for key, value in adf[4].items():
      print('\\t{}: {}'.format(key, value))
    
    ```
    
- 눈으로 확인하기: mean, std가 constant over time한지, 패턴이나 추세가 보이는 지?
![[Pasted image 20231218174239.png]]
    
    ```python
    plt.plot(wn, label='white noise')
    plt.plot(wn.rolling(window=20).mean(), label='rolling_mean')  # 내 앞 20개의 평균값들
    plt.plot(wn.rolling(window=20).std(), label='rolling_std')  # 내 앞 20개의 분산값들
    plt.legend()
    plt.show()
    
    ```
    

### AR, MR, ARMA

- $\epsilon$ 은 전부 white noise여야 함
AR(p) model: $y_t = \alpha_0 + \alpha_1 y_{t-1} + \alpha_2 y_{t-2} + \cdots + \alpha_p y_{t-p}$
MA(q) model: $y_t = \epsilon_t + \beta_1 \epsilon_{t-1} + \beta_2 \epsilon_{t-2} + \cdots + \beta_q \epsilon_{t-q}$
ARMA(p,q) model: $y_t = \alpha_0 + \alpha_1 y_{t-1} + \alpha_2 y_{t-2} + \cdots + \alpha_p y_{t-p} + \epsilon_t + \beta_1 \epsilon_{t-1} + \beta_2 \epsilon_{t-2} + \cdots + \beta_q \epsilon_{t-q}$

$f(x) \simeq \hat{f}(x) \iff f(x)=\hat{f}(x)+\epsilon(x)$
$\epsilon(x) = f(x) - \hat{f}(x)$

### Auto Arima

- auto arima로 찾은 값을 base로 잡아서 이것보다 높이는 것이 의미가 있

### Box-Cox Transformation, Log Transformation

- 분산이 요동치는 것을 잡아주기 위함 constant variance
- 나중에 metrics 볼때나, prediction/target 비교할 때에는 꼭 inverse transform 해주어야 함
- 2^n, 4^n, 6^n
    - n = 1.2

---

### Stores Dataset 넣고 돌려보기

- 시계열에서 빈칸이 있으면 안 됨!
    - 일 단위라면 range내에 모든 date가 다 존재해야 함.