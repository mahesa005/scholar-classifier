import numpy as np
import pandas as pd
import pickle  # <--- PENTING: Library untuk menyimpan file
import os      # <--- PENTING: Library untuk path folder
from collections import Counter

class Node:
    def __init__(self, prediction=None):
        self.prediction = prediction
        self.feature_idx = None
        self.feature_name = None
        self.is_continuous = False
        self.threshold = None
        self.children = {}

class ID3DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None
        self.feature_types = []
        self.feature_names = []

    # --- FITUR BARU: SAVE & LOAD ---
    def save_model(self, filename):
        """Menyimpan objek model ke file .pkl"""
        # Pastikan folder tujuan ada
        folder = os.path.dirname(filename)
        if folder and not os.path.exists(folder):
            try:
                os.makedirs(folder)
                print(f"📁 Membuat folder baru: {folder}")
            except OSError as e:
                print(f"❌ Gagal membuat folder {folder}: {e}")
                return

        try:
            with open(filename, 'wb') as f:
                pickle.dump(self, f)
            print(f"💾 Model BERHASIL disimpan ke: {filename}")
        except Exception as e:
            print(f"❌ Gagal menyimpan model: {e}")

    @staticmethod
    def load_model(filename):
        """Memuat model dari file .pkl"""
        if not os.path.exists(filename):
            print(f"❌ File model tidak ditemukan: {filename}")
            return None
        try:
            with open(filename, 'rb') as f:
                model = pickle.load(f)
            print(f"📂 Model berhasil dimuat dari: {filename}")
            return model
        except Exception as e:
            print(f"❌ Gagal memuat model: {e}")
            return None

    # --- LOGIKA UTAMA (ID3) ---
    def fit(self, X, y, feature_names=None):
        X = pd.DataFrame(X).copy()
        y = np.array(y)

        # Data Cleaning (Imputation)
        for col in X.columns:
            if X[col].isnull().any():
                if pd.api.types.is_numeric_dtype(X[col]):
                    X[col] = X[col].fillna(X[col].mean())
                else:
                    X[col] = X[col].fillna(X[col].mode()[0])

        X = X.values

        self.feature_types = []
        n_features = X.shape[1]
        for i in range(n_features):
            if isinstance(X[0, i], (int, float, np.number)) and not isinstance(X[0, i], str):
                self.feature_types.append('continuous')
            else:
                self.feature_types.append('categorical')

        if feature_names is None:
            self.feature_names = [f"feat_{i}" for i in range(n_features)]
        else:
            self.feature_names = feature_names

        attribute_indices = list(range(n_features))
        self.root = self._id3(X, y, attribute_indices, depth=0)

    def _id3(self, X, y, attribute_indices, depth):
        n_samples = X.shape[0]
        n_labels = len(np.unique(y))

        if n_labels == 1: return Node(prediction=y[0])
        if len(attribute_indices) == 0 or depth >= self.max_depth or n_samples < self.min_samples_split:
            return Node(prediction=self._most_common_label(y))

        best_feat_idx, best_gain, best_threshold = self._get_best_attribute(X, y, attribute_indices)

        if best_feat_idx is None: return Node(prediction=self._most_common_label(y))

        node = Node()
        node.feature_idx = best_feat_idx
        node.feature_name = self.feature_names[best_feat_idx]
        node.is_continuous = (self.feature_types[best_feat_idx] == 'continuous')

        if node.is_continuous:
            node.threshold = best_threshold
            left_mask = X[:, best_feat_idx] <= best_threshold
            
            X_left, y_left = X[left_mask], y[left_mask]
            X_right, y_right = X[~left_mask], y[~left_mask]
            
            if len(X_left) == 0 or len(X_right) == 0: return Node(prediction=self._most_common_label(y))

            node.children['<='] = self._id3(X_left, y_left, attribute_indices, depth + 1)
            node.children['>'] = self._id3(X_right, y_right, attribute_indices, depth + 1)

        else:
            unique_values = np.unique(X[:, best_feat_idx])
            new_attributes = [a for a in attribute_indices if a != best_feat_idx]
            
            for val in unique_values:
                mask = X[:, best_feat_idx] == val
                X_subset, y_subset = X[mask], y[mask]
                
                if len(X_subset) == 0: child = Node(prediction=self._most_common_label(y))
                else: child = self._id3(X_subset, y_subset, new_attributes, depth + 1)
                node.children[val] = child

        return node

    def _get_best_attribute(self, X, y, attribute_indices):
        best_gain = -1
        best_feat_idx = None
        best_threshold = None

        for feat_idx in attribute_indices:
            X_col = X[:, feat_idx]
            is_cont = (self.feature_types[feat_idx] == 'continuous')
            
            if is_cont:
                thresholds = np.unique(X_col)
                if len(thresholds) > 100:
                    thresholds = np.percentile(thresholds, np.linspace(0, 100, 10))
                
                for thr in thresholds:
                    gain = self._calculate_gain_continuous(y, X_col, thr)
                    if gain > best_gain:
                        best_gain = gain; best_feat_idx = feat_idx; best_threshold = thr
            else:
                gain = self._calculate_gain_categorical(y, X_col)
                if gain > best_gain:
                    best_gain = gain; best_feat_idx = feat_idx; best_threshold = None

        return best_feat_idx, best_gain, best_threshold

    def _calculate_gain_categorical(self, y, X_col):
        parent_entropy = self._entropy(y)
        n = len(y)
        child_entropy_sum = 0
        unique_vals, counts = np.unique(X_col, return_counts=True)
        for val, count in zip(unique_vals, counts):
            child_entropy_sum += (count / n) * self._entropy(y[X_col == val])
        return parent_entropy - child_entropy_sum

    def _calculate_gain_continuous(self, y, X_col, threshold):
        parent_entropy = self._entropy(y)
        n = len(y)
        left_mask = X_col <= threshold
        if np.sum(left_mask) == 0 or np.sum(~left_mask) == 0: return 0
        
        y_left, y_right = y[left_mask], y[~left_mask]
        child_entropy = (len(y_left)/n)*self._entropy(y_left) + (len(y_right)/n)*self._entropy(y_right)
        return parent_entropy - child_entropy

    def _entropy(self, y):
        hist = np.bincount(y) if np.issubdtype(y.dtype, np.integer) else np.unique(y, return_counts=True)[1]
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def _most_common_label(self, y):
        if len(y) == 0: return None
        return Counter(y).most_common(1)[0][0]

    def predict(self, X):
        X = np.array(X, dtype=object)
        return np.array([self._traverse(x, self.root) for x in X])

    def _traverse(self, x, node):
        if node.prediction is not None: return node.prediction
        val = x[node.feature_idx]
        if node.is_continuous:
            if val <= node.threshold:
                if '<=' in node.children: return self._traverse(x, node.children['<='])
            else:
                if '>' in node.children: return self._traverse(x, node.children['>'])
        else:
            if val in node.children: return self._traverse(x, node.children[val])
        
        # Fallback aman
        if len(node.children) > 0:
            return self._traverse(x, list(node.children.values())[0])
        return node.prediction

    def print_tree(self, node=None, prefix="", is_last=True, is_root=True, branch_label="", max_depth=3, current_depth=0):
        if node is None: node = self.root
        if current_depth > max_depth:
            if is_last and not is_root: print(f"{prefix}└── ...")
            return

        if node.prediction is not None:
            display_txt = f"\033[92mOutput: {node.prediction}\033[0m"
        else:
            ft_name = node.feature_name if node.feature_name else f"Feat{node.feature_idx}"
            display_txt = f"\033[94m[{ft_name}?]\033[0m"

        if is_root:
            print(display_txt); new_prefix = ""
        else:
            connector = "└── " if is_last else "├── "
            print(f"{prefix}{connector}\033[93m({branch_label})\033[0m {display_txt}")
            new_prefix = prefix + ("    " if is_last else "│   ")

        if node.prediction is None:
            children_keys = list(node.children.keys())
            for i, key in enumerate(children_keys):
                self.print_tree(node.children[key], new_prefix, (i == len(children_keys) - 1), False, str(key), max_depth, current_depth + 1)

# --- BAGIAN UTAMA (YANG BERUBAH) ---
if __name__ == "__main__":
    import sys
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    print("🚀 Memulai ID3 pada Dataset Tugas Besar (Real Data)...")

    # 1. SETUP PATH 
    base_dir = os.path.dirname(os.path.abspath(__file__)) 
    project_root = os.path.dirname(base_dir)              
    data_path = os.path.join(project_root, 'data', 'train.csv')
    
    # Path Folder Models
    models_dir = os.path.join(project_root, 'models')
    model_save_path = os.path.join(models_dir, 'dtl_model.pkl')

    print(f"📂 Mencari dataset di: {data_path}")

    try:
        # 2. Load & Split
        df = pd.read_csv(data_path)
        if 'id' in df.columns: df = df.drop('id', axis=1)
        
        print(f"✅ Berhasil Load Data! Ukuran: {df.shape}")
        
        X = df.drop('Target', axis=1).values
        y = df['Target'].values
        feature_names = df.drop('Target', axis=1).columns.tolist()
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # 3. Training
        print("\n🌳 Sedang melatih pohon ID3 (Mungkin butuh waktu)...")
        model = ID3DecisionTree(max_depth=5, min_samples_split=10)
        model.fit(X_train, y_train, feature_names=feature_names)
        print("✅ Training Selesai!")

        # 4. Evaluasi
        print("\n🧪 Melakukan Prediksi pada Data Validasi...")
        acc = accuracy_score(y_val, model.predict(X_val))
        print(f"🎯 Akurasi Validasi: {acc * 100:.2f}%")

        # 5. Visualisasi
        print("\n🌳 Struktur Pohon (Top 3 Levels):")
        model.print_tree(max_depth=3)

        # 6. SAVE MODEL (Ini yang sebelumnya hilang)
        print(f"\n💾 Menyimpan model ke: {model_save_path}")
        model.save_model(model_save_path)

        # 7. TEST LOAD
        print("♻️  Mencoba load model kembali untuk verifikasi...")
        loaded_model = ID3DecisionTree.load_model(model_save_path)
        if loaded_model:
            acc_loaded = accuracy_score(y_val, loaded_model.predict(X_val))
            print(f"✅ Model berhasil di-load! Akurasi Load: {acc_loaded * 100:.2f}%")

    except FileNotFoundError:
        print(f"❌ ERROR: File tidak ditemukan di: {data_path}")
    except Exception as e:
        print(f"❌ Terjadi Error: {e}")