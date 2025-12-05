from abc import ABC, abstractmethod
import numpy as np
from src.utils.metrics import f1_macro
import os   
import pickle
import tempfile
import shutil

class BaseClassifier(ABC):
   
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        pass
    
    @abstractmethod
    def predict(self, X:np.ndarray) -> np.ndarray:
        pass

    def save_model(self, filename):
        """Save model using a temporary file to avoid corruption"""
        folder = os.path.dirname(filename)
        if folder and not os.path.exists(folder):
            try: 
                os.makedirs(folder)
            except OSError: 
                pass
        
        # Create a temporary file in the same directory
        temp_fd, temp_path = tempfile.mkstemp(dir=folder if folder else '.', suffix='.tmp')
        
        try:
            # Write to temporary file first
            with os.fdopen(temp_fd, 'wb') as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Only replace the target file if pickle succeeded
            shutil.move(temp_path, filename)
            print(f"Model BERHASIL disimpan ke: {filename}")
            return True
            
        except Exception as e:
            # Clean up temp file on error
            try:
                os.remove(temp_path)
            except:
                pass
            print(f"Gagal menyimpan model: {e}")
            return False

    @staticmethod
    def load_model(filename):
        """Load model from file"""
        if not os.path.exists(filename): 
            print(f"File tidak ditemukan: {filename}")
            return None
        
        # Check if file is empty
        if os.path.getsize(filename) == 0:
            print(f"File kosong: {filename}")
            return None
            
        try:
            with open(filename, 'rb') as f: 
                return pickle.load(f)
        except Exception as e: 
            print(f"Gagal memuat model: {e}")
            return None


    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        # Calculate F1 macro score
        y_pred = self.predict(X)
        return f1_macro(y, y_pred)