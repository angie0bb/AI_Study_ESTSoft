# Polynomial Model, Gradient Descent

태그: Polynomial, SGD, 경사하강법
No.: 4

# Model

- 파라미터 $\theta = \{w1, w2, \dots\}$ 로 구성된 mathematical function $f_{\theta}$
- 세 가지 요소 필요
    - 함수 $f$
    - 파라미터 세팅 및 가져오기 위한 방식
    - batch를 처리하기 위한 방식
- code example
    
    ```python
    class Module:
      def __init__(self) -> None:
        raise NotImplementedError
    
      def set_params(self) -> None:
        raise NotImplementedError
      
      def get_params(self) -> dict:
        raise NotImplementedError
      
      def forward(self, x:np.array) -> np.array:
        raise NotImplementedError
      
      def __call__(self, x:np.array) -> np.array:
        return self.forward(x)
    
    class LinearModel(Module):
      def __init__(self, w:float=.0, b:float=.0) -> None:
        self.set_params(w, b)
    
      def set_params(self, w:float, b:float) -> None:
        self.w = w
        self.b = b
    
      def get_params(self) -> dict[str,float]:
        return {'w': self.w, 'b':self.b}
    
      def forward(self, x:np.array) -> np.array:
        params = self.get_params()
        w = params.get('w')
        b = params.get('b')
        return w * x + b
    ```
    

# Gradient of MSE

## Gradient

- 전제
    - $f_\theta(x) = x w + b, \text{where }\theta=(w,b)$
    - $\operatorname{MSE}_\theta(x, y) = \frac{1}{n}\sum^n_{i=1}(f(x_i) - y_i)^2 = \frac{1}{n}\sum^n_{i=1}(x_iw+b - y_i)^2$
- gradient of MSE with $f_\theta(x)$
    
    $$
    \frac{\partial}{\partial w} \operatorname{MSE}_{\theta}(x,y) = \frac{2}{n}\sum^n_{i=1}x_i(x_iw+b - y_i)=\frac{2}{n}\sum^n_{i=1}x_i(\hat y_i - y_i).
    
    $$
    

$$
\frac{\partial}{\partial b} \operatorname{MSE}{\theta}(x,y) = \frac{2}{n}\sum^n{i=1}(x_iw+b - y_i)= \frac{2}{n}\sum^n_{i=1}(\hat y_i - y_i).
$$

- 미분 관련 추가 정보 ($f,g:\mathbb R\rightarrow \mathbb R$)
    - $f(x)=c \implies f'(x)=0$
    - $f(x)=ax+b \implies f'(x)=a$
    - $f(x)=ax^2 + bx + c \implies f'(x)=2ax + b$
    - $(f(x) + g(x))' = f'(x) + g'(x)$
    - $**(f(x)g(x))' = f'(x)g(x) + f(x)g'(x)**$
    - $(f(g(x)))' = f'(g(x))g'(x)$
    - $\frac{d(x^2)}{dx} = (x^2)' = 2x$
- code example
    
    ```python
    def mse(y_hat:np.array, y_true:np.array) -> float:
      assert len(y_hat) == len(y_true)
      return ((y_hat - y_true)**2).mean()
    
    def grad_mse(model:Module, x:np.array, y_true:np.array) -> dict[str,float]:
      assert len(x) == len(y_true)
      n = len(x)
      y_hat = model(x)
      d_w = 2*(x*(y_hat-y_true)).mean()
      d_b = 2*(y_hat-y_true).mean()
      return {'d_w': d_w, 'd_b':d_b}
    ```
    

## Gradient update with lr

$$
\theta_{\textrm{new}} = \theta_{\textrm{old}} - \alpha \nabla\operatorname{MSE}_{\theta_{\textrm{old}}}(x,y), \text{ where }\alpha>0\text{ is learning rate}

$$

- code example
    
    ```python
    def update(model:Module, lr:float, d_w:float, d_b:float) -> None:
      params_old = model.get_params()
      params_new = {
        'w': params_old.get('w') - lr*d_w,
        'b': params_old.get('b') - lr*d_b,
      }
      model.set_params(**params_new)
    
    lin2 = LinearModel(0,0)
    grad_mse(lin2, xs, ys)
    
    history = [lin2.get_params()]
    
    for epoch in range(200):
      grad = grad_mse(lin2, xs, ys)
      update(lin2, 0.2, **grad)
      err = mse(lin2(xs), ys)
      params = lin2.get_params()
      history.append(params)
      print(f"Epoch {epoch+1}: mse={err:.4f}, w={params.get('w'):.4f}, b={params.get('b'):.4f}")
    ```
    

## Gradient Descent

- gradient descent: 가중치($a_1^{(1)}, b_1^{(1)}, ...$)를 최적화하여 목표 함수에 가장 가까운 함수가 되도록 만든다.⇒ 학습
    - 최적의 가중치는 유일하지 않을 수 있음. 학습을 통해 가장 근사화할 수 있는 값들을 결정해야 함
    - 모델이 깊고 넓어지면서 설정해야 할 파라미터 수가 많아짐 → “정확한”함수를 찾는 것이 아닌 “근사화”에 집중

## Visualization

- code example
    
    ```python
    import plotly.express as px
    import pandas as pd
    
    df = pd.DataFrame(history, columns=['w','b'])
    df = df.set_index(df.index.set_names('epoch')).reset_index()
    df0 = df.copy()
    df1 = df.copy()
    df0['x'] = xs.min()
    df1['x'] = xs.max()
    df = pd.concat([df0, df1]).reset_index(drop=True)
    df['y'] = df.w * df.x + df.b
    
    fig = px.line(df, x='x', y='y', animation_frame="epoch", width=500, height=500)
    
    fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 0.1
    fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 0.1
    fig.layout.updatemenus[0].buttons[0].args[1]['frame']['redraw'] = True
    
    fig.add_scatter(x=xs.flatten(), y=ys.flatten(), mode='markers', name='data', marker={'size':2})
    
    for i, frame in enumerate(fig.frames):
        frame['layout']['title_text'] = f"Prediction: y = {history[i]['w']:.4f}x{'' if history[i]['b'] < 0 else '+'}{history[i]['b']:.4f}"
    
    fig.update_layout(template='plotly_dark')
    fig.show()
    ```
    

# Exercise

- 위 모델 class / mse 함수 코드를 기반으로 polynomial(2차식) model 학습
    - 각 기능을 분리화하여 구조화할 것
    - 각 class 및 function에 docstring을 추가할 것
    - polynomial에 맞게 `grad_mse()` 내부를 수정할 것

## Result

- `model.py`
    
    ```python
    import numpy as np
    
    class Module:
        """
        Basic module for model implementation
        """
        def __init__(self) -> None:
            """__init__
            class를 만들 때 최초 초기화 작업
            Args:
                self: 객체의 인스턴스 그 자체 (주소값이 같다)
    
            Raises:
                ValueError: 인자가 조건에 맞지 않는 경우
    
            Returns:
                NotImplementedError : 추가 기능을 만들 때 사용되는 예외처리(자식클래스에서 구현)
            """
            raise NotImplementedError
    
        def set_params(self) -> None:
            raise NotImplementedError
        
        def get_params(self) -> dict:
            raise NotImplementedError
        
        def forward(self, x:np.array) -> np.array:
            raise NotImplementedError
        
        def __call__(self, x:np.array) -> np.array:
            """__call__
            __call__ = forward를 사용해도 되지만 현재 코드에서 
            __call__사용시 forward함수가 불러와 질 수 있기 때문에 
            __call__함수를 활용
            Args:
            self: 객체의 인스턴스 그 자체 (주소값이 같다)
            x(np.array): x 데이터셋
    
            Raises:
            ValueError: 인자가 조건에 맞지 않는 경우
    
            Returns:
            self.forward : x입력을 y_hat(예측값)
            """
            return self.forward(x)
      
    
    class LinearModel(Module):
        """
        Linear model: y = wx + b
        """
        def __init__(self, w:float=.0, b:float=.0) -> None:
            """__init__
            Args:
            w(float) = 가중치(기울기)
            b(float) = 바이어스(상수항)
    
            Raises:
            ValueError: 인자가 조건에 맞지 않는 경우
            Returns:
            """
            self.set_params(w, b)
    
        def set_params(self, w:float, b:float) -> None:
            """set_params
            Set parameter for linear model
            2개를 받아 똑같은 이름으로 변수 저장
            Args:
                w(float): coefficient of x
                b(float): constant
            Raises:
            ValueError: 인자가 조건에 맞지 않는 경우
    
            Returns:
            """
            self.w = w
            self.b = b
    
        def get_params(self) -> dict[str,float]:
            """get_params
            Get parameter of linear model
            저장 된 변수 딕셔너리 형태로
            Raises:
            ValueError: 인자가 조건에 맞지 않는 경우
            Return:
                dictionary of w, b
            """
            return {'w': self.w, 'b':self.b}
    
        def forward(self, x:np.array) -> np.array:
            """forward
            Forward function
            가져온 값을 바탕으로 계산
            Args:
                x(array): dictionary of w,b
            Return:
                dictionary of w, b
                함수f(poly Linear)
            """
            params = self.get_params()
            w = params.get('w')
            b = params.get('b')
            return w * x + b
    
    class PolynomialModel(Module): # Polynomial
        """class
        다항식 
        Args:
            Module: 부모클래스
        """
        # Initialize Polynomial model (y = w1_x^2 + w2_x + b) 되도록이면 계수를 맞춰주기 w2: x^2 
        def __init__(self, w1:float=.0, w2:float=.0, b:float=.0) -> None:
            """__init__
            Args:
                w1(float) = 가중치(기울기)
                w2(float) = 가중치(기울기)
                b(float) = 바이어스(상수항)
    
            Raises:
                ValueError: 인자가 조건에 맞지 않는 경우
            Returns:
            """
            self.set_params(w1,w2,b)
    
        def set_params(self, w1:float, w2:float, b:float) -> None:
            """set_params
            Set parameter for linear model
            3개를 받아 똑같은 이름으로 변수 저장
            Args:
                w1(float): coefficient of x^2
                w2(float): coefficient of x
                b(float): constant
            Raises:
            ValueError: 
                인자가 조건에 맞지 않는 경우
            """
            self.w1 = w1
            self.w2 = w2
            self.b = b
    
        def get_params(self) -> dict[str,float]:
            """get_params
            Get parameter of linear model
            저장 된 변수 딕셔너리 형태로
            Raises:
                ?
            ValueError: 
                인자가 조건에 맞지 않는 경우
            Return:
                dictionary of w1, w2, b
            """
            return {'w1': self.w1, 'w2': self.w2, 'b':self.b}
    
        def forward(self, x:np.array) -> np.array:
            params = self.get_params()
            w1 = params.get('w1')
            w2 = params.get('w2')
            b = params.get('b')
            return w1 * (x**2) + w2*x  + b
    ```
    
- `loss.py`
    
    ```python
    from model import Module
    import numpy as np
    
    class MSE():
        def __init__(self):
            """set_params
            MSE 공식
            Args:
    
            Raises:
                ValueError: 인자가 조건에 맞지 않는 경우
    
            Returns:
                MSE
                y_hat,y_true = ((y_hat - y_true)**2)의 평균
            """
            self.mse_func = lambda y_hat, y_true: ((y_hat - y_true)**2).mean()
    
        def get_mse(self, y_hat:np.array, y_true:np.array) -> float:
            """get_mse
            MSE 받아오기
            Args:
                y_hat(array): 예측값
                y_true(array): 실제값
            Raises:
                ValueError: 인자가 조건에 맞지 않는 경우
    
            Returns:
                assert: y_hat 과 y_true값이 같아 질 때 종료
                mse값
            """
            assert len(y_hat) == len(y_true)
            return self.mse_func(y_hat, y_true)
    
        def get_grad_mse(self, model:Module, x:np.array, y_true:np.array) -> dict[str,float]:  
            """get_grad_mse
            MSE 받아오기
            Args:
                model():class model(부모클래스)를 상속
                x(array): 예측값
                y_true(array): 실제값
            Raises:
                ValueError: 인자가 조건에 맞지 않는 경우
    
            Returns:
                assert: y_hat 과 y_true값이 같아 질 때 종료
                mse값
                d_w: 편미분w값(기울기)
                d_b: 편미분b값(기울기)
            """
            assert len(x) == len(y_true)
            n = len(x)
            y_hat = model(x)
            d_w = 2*(x*(y_hat-y_true)).mean()
            d_b = 2*(y_hat-y_true).mean()
            return {'d_w': d_w, 'd_b':d_b}
        
        def update(self, model:Module, lr:float, d_w:float, d_b:float) -> None:
            params_old = model.get_params()
            params_new = {
                'w': params_old.get('w') - lr*d_w,
                'b': params_old.get('b') - lr*d_b,
            }
            model.set_params(**params_new)
    
    class PolyMSE(MSE):
        def __init__(self):
            """set_params
            MSE 공식
            Args:
                parameter로 들어갈 것들
            Raises: 
                ValueError: 인자가 조건에 맞지 않는 경우
    
            Returns:
                MSE
                y_hat,y_true = ((y_hat - y_true)**2)의 평균
            """
            self.mse_func = lambda y_hat, y_true: ((y_hat - y_true)**2).mean()
    
        def get_grad_mse(self, model:Module, x:np.array, y_true:np.array) -> dict[str,float]:
            """get_grad_mse
            MSE 받아오기
            Args:
                model(Module):class model(부모클래스)를 상속 ############3
                x(array): 예측값
                y_true(array): 실제값
            Raises:
                ValueError: 인자가 조건에 맞지 않는 경우
    
            Returns:
                assert: y_hat 과 y_true값이 같아 질 때 종료
                mse값
                d_w1: 편미분w값(기울기)
                d_w2: 편미분w값(기울기)
                d_b: 편미분b값(기울기)
            """
            assert len(x) == len(y_true)
            n = len(x)
            y_hat = model(x)
            d_w1 = 2*((x**2)*(y_hat-y_true)).mean() 
            d_w2 = 2*(x*(y_hat-y_true)).mean()
            d_b = 2*(y_hat-y_true).mean()
            return {'d_w1': d_w1, 'd_w2':d_w2, 'd_b':d_b}
    
        def update(self, model:Module, lr:float, d_w1:float, d_w2:float, d_b:float) -> None:
            """update
            MSE 받아오기
            Args:
                model: 부모클래스 상속
                lr: learning rate(학습률, 기울기 변환속도)
                d_w1: 편미분w값(기울기)
                d_w2: 편미분w값(기울기)
                d_b: 편미분b값(기울기)
            Raises:
                ValueError: 인자가 조건에 맞지 않는 경우
            """
            params_old = model.get_params()
            params_new = {
                'w1': params_old.get('w1') - lr*d_w1,
                'w2': params_old.get('w2') - lr*d_w2,
                'b': params_old.get('b') - lr*d_b,
            }
            model.set_params(**params_new)
    ```
    
- `main.ipynb`
    
    ```python
    import numpy as np
    import matplotlib.pyplot as plt
    from model import *
    from loss import *
    
    f = lambda x: 3*x**2 + 2*x+ 1.0
    
    xs = np.random.rand(1000, 1)   # 1000 points
    ys = f(xs) + 0.1*np.random.randn(1000, 1)
    
    plt.title("Dataset")
    plt.scatter(xs, ys, s=1)
    plt.legend()
    plt.show()
    
    poly = PolynomialModel()
    mse = PolyMSE()
    mse.get_grad_mse(model=poly, x=xs, y_true=ys)
    
    history = [poly.get_params()]
    
    count = 0
    prev_err = 0
    early_stop_criterion = 0  #리소스 절약용 -> 원래는 overfitting을 방지하기 위해 사용
    # early stop 조건이 너무 커서, 예상값과 차이가 벌어짐. 
    for epoch in range(5000): # epoch 값이 작은 것도 영향을 미침. 
      grad = mse.get_grad_mse(poly, xs, ys)
      mse.update(poly, 0.2, **grad)
      err = mse.get_mse(poly(xs), ys)
      if abs(err - prev_err) < early_stop_criterion:
        count += 1
      else:
        count = 0
      prev_err = err
      if count == 3:
        print(f"Early stop with epoch {epoch}")
        print(f"Last status: mse={err:.4f}, w1={params.get('w1'):.4f}, w2={params.get('w2'):.4f}, b={params.get('b'):.4f}")
        break
      params = poly.get_params()
      history.append(params)
      print(f"Epoch {epoch+1}: mse={err:.4f}, w1={params.get('w1'):.4f}, w2={params.get('w2'):.4f}, b={params.get('b'):.4f}")
    ```