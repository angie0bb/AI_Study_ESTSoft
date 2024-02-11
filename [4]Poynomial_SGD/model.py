import numpy as np

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
  


class PolynomialModel(Module): # Polynomial

  # Initialize Polynomial model (y = w1_x^2 + w2_x + b)
  def __init__(self, w1:float=.0, w2:float=.0, b:float=.0) -> None:
    self.set_params(w1,w2,b)

  def set_params(self, w1:float, w2:float, b:float) -> None:
    self.w1 = w1
    self.w2 = w2
    self.b = b

  def get_params(self) -> dict[str,float]:
    return {'w1': self.w1, 'w2': self.w2, 'b':self.b}

  def forward(self, x:np.array) -> np.array:
    params = self.get_params()
    w1 = params.get('w1')
    w2 = params.get('w2')
    b = params.get('b')
    return w1 * (x**2) + w2*x  + b