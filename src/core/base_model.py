from abc import ABC, abstractmethod
import numpy as np
from utils.metrics import f1_macro

class BaseClassifier(ABC):
   
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        pass
    
    @abstractmethod
    def predict(self, X:np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        # Calculate F1 macro score
        y_pred = self.predict(X)
        return f1_macro(y, y_pred)