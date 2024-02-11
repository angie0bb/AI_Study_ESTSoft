from .model import Module
import numpy as np

class MSE():
    def __init__(self):
        self.mse_func = lambda y_hat, y_true: ((y_hat - y_true)**2).mean()

    def get_mse(self, y_hat:np.array, y_true:np.array) -> float:
        assert len(y_hat) == len(y_true)
        return self.mse_func(y_hat, y_true)

    def grad_mse(model:Module, x:np.array, y_true:np.array) -> dict[str,float]:
        assert len(x) == len(y_true)
        n = len(x)
        y_hat = model(x)
        d_w1 = 2*((x**2)*(y_hat-y_true)).mean() 
        d_w2 = 2*(x*(y_hat-y_true)).mean()
        d_b = 2*(y_hat-y_true).mean()
        return {'d_w1': d_w1, 'd_w2':d_w2, 'd_b':d_b}