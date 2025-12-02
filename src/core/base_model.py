from abc import ABC, abstractmethod
import numpy as np
from utils.metrics import f1_macro
import os   
import pickle

class BaseClassifier(ABC):
   
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        pass
    
    @abstractmethod
    def predict(self, X:np.ndarray) -> np.ndarray:
        pass

    def save_model(self, filename):
        folder = os.path.dirname(filename)
        if folder and not os.path.exists(folder):
            try: os.makedirs(folder)
            except OSError: pass
        try:
            with open(filename, 'wb') as f: pickle.dump(self, f)
            print(f"Model BERHASIL disimpan ke: {filename}")
        except Exception as e: print(f"Gagal menyimpan model: {e}")

    @staticmethod
    def load_model(filename):
        if not os.path.exists(filename): return None
        try:
            with open(filename, 'rb') as f: return pickle.load(f)
        except Exception as e: print(f"Gagal memuat model: {e}"); return None

        

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        # Calculate F1 macro score
        y_pred = self.predict(X)
        return f1_macro(y, y_pred)