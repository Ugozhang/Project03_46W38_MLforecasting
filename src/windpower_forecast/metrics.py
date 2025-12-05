# metrics.py
import numpy as np

class ForecastEvaluator:
    def __init__(self, y_true, y_pred):
        self.y_true = np.asarray(y_true)
        self.y_pred = np.asarray(y_pred)

    def mse(self):
        return np.mean((self.y_true - self.y_pred)**2)

    def mae(self):
        return np.mean(np.abs(self.y_true - self.y_pred))

    def rmse(self):
        return np.sqrt(self.mse())
