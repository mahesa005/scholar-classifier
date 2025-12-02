import numpy as np
import pandas as pd
import os
from collections import Counter

from core.base_model import BaseClassifier

# --- 1. CLASS NODE ---
class Node:
    def __init__(self, prediction=None):
        self.prediction = prediction
        self.feature_idx = None
        self.feature_name = None
        self.is_continuous = False
        self.threshold = None
        self.children = {}

# --- 2. CLASS ID3 DECISION TREE ---
class ID3DecisionTree(BaseClassifier):
    def __init__(self, min_samples_split=2, max_depth=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None
        self.feature_types = []
        self.feature_names = []

    # --- TRAINING ---
    def fit(self, X, y, feature_names=None):

        # 1. PERSIAPAN DATA (Data Preprocessing)
        X = pd.DataFrame(X).copy()
        y = np.array(y)
        # 2. MEMBERSIHKAN DATA (Handling Missing Values)
        for col in X.columns:
            if X[col].isnull().any():
                if pd.api.types.is_numeric_dtype(X[col]): X[col] = X[col].fillna(X[col].mean())
                else: X[col] = X[col].fillna(X[col].mode()[0])
        X = X.values
        # 3. KENALAN DENGAN FITUR (Type Detection)
        self.feature_types = []
        n_features = X.shape[1]
        for i in range(n_features):
            if isinstance(X[0, i], (int, float, np.number)) and not isinstance(X[0, i], str):
                self.feature_types.append('continuous')
            else: self.feature_types.append('categorical')

        # 4. MULAI MEMBANGUN POHON (Core Algorithm)
        self.feature_names = feature_names if feature_names else [f"feat_{i}" for i in range(n_features)]
        attribute_indices = list(range(n_features))
        self.root = self._id3(X, y, attribute_indices, depth=0)

    # --- LOGIKA ID3 ---
    def _id3(self, X, y, attribute_indices, depth):
        n_samples = X.shape[0]
        n_labels = len(np.unique(y))

        # Jika semua data labelnya sama (Murni) -> Berhenti
        if n_labels == 1: return Node(prediction=y[0])
        
        # Cek Max Depth (Rem Darurat)
        hit_max_depth = (self.max_depth is not None and depth >= self.max_depth)

         # Jika fitur habis, atau sampel terlalu sedikit -> Berhenti (Voting terbanyak)
        if len(attribute_indices) == 0 or hit_max_depth or n_samples < self.min_samples_split:
            return Node(prediction=self._most_common_label(y))

       # Panggil fungsi matematika (Entropy & Gain)
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
        best_gain = -1; best_feat_idx = None; best_threshold = None
        for feat_idx in attribute_indices:
            X_col = X[:, feat_idx]
            is_cont = (self.feature_types[feat_idx] == 'continuous')
            if is_cont:
                thresholds = np.unique(X_col)
                if len(thresholds) > 50: thresholds = np.percentile(thresholds, np.linspace(0, 100, 10))
                for thr in thresholds:
                    gain = self._calculate_gain_continuous(y, X_col, thr)
                    if gain > best_gain: best_gain = gain; best_feat_idx = feat_idx; best_threshold = thr
            else:
                gain = self._calculate_gain_categorical(y, X_col)
                if gain > best_gain: best_gain = gain; best_feat_idx = feat_idx; best_threshold = None
        return best_feat_idx, best_gain, best_threshold

    def _calculate_gain_categorical(self, y, X_col):
        parent_entropy = self._entropy(y)
        n = len(y); child_entropy_sum = 0
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
        if len(node.children) > 0: return self._traverse(x, list(node.children.values())[0])
        return node.prediction

    # --- VISUALISASI (UPDATED: TERIMA PARAMETER FILE) ---
    def print_tree(self, node=None, prefix="", is_last=True, is_root=True, branch_label="", max_depth=None, current_depth=0, file=None):
        if node is None: node = self.root
        
        # Cek apakah print ke Layar (Warna) atau File (Polos)
        use_color = (file is None)

        if max_depth is not None and current_depth > max_depth:
            if is_last and not is_root: print(f"{prefix}└── ...", file=file)
            return

        if node.prediction is not None:
            txt = f"Output: {node.prediction}"
            if use_color: txt = f"\033[92m{txt}\033[0m"
        else:
            ft_name = node.feature_name if node.feature_name else f"Feat{node.feature_idx}"
            txt = f"[{ft_name}?]"
            if use_color: txt = f"\033[94m{txt}\033[0m"

        if is_root:
            print(txt, file=file)
            new_prefix = ""
        else:
            connector = "└── " if is_last else "├── "
            branch_txt = f"({branch_label})"
            if use_color: branch_txt = f"\033[93m{branch_txt}\033[0m"
            print(f"{prefix}{connector}{branch_txt} {txt}", file=file)
            new_prefix = prefix + ("    " if is_last else "│   ")

        if node.prediction is None:
            children_keys = list(node.children.keys())
            for i, key in enumerate(children_keys):
                self.print_tree(node.children[key], new_prefix, (i == len(children_keys) - 1), False, str(key), max_depth, current_depth + 1, file=file)


# --- MAIN PROGRAM (FINAL + TXT EXPORT) ---
if __name__ == "__main__":
    import sys
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.tree import DecisionTreeClassifier

    print("🚀 Memulai Program ID3 Decision Tree...")

    # 1. Setup Path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    train_path = os.path.join(project_root, 'data', 'train.csv')
    test_path  = os.path.join(project_root, 'data', 'test.csv')
    
    # Output Files
    models_dir = os.path.join(project_root, 'models')
    if not os.path.exists(models_dir): os.makedirs(models_dir)
    
    save_model_path = os.path.join(models_dir, 'dtl_model.pkl')
    save_txt_path = os.path.join(models_dir, 'dtl_model.txt') # <--- Path untuk file TXT
    
    sub_scratch_path = os.path.join(project_root, 'data', 'submission_scratch.csv')
    sub_sklearn_path = os.path.join(project_root, 'data', 'submission_sklearn.csv')

    try:
        # --- INPUT USER ---
        print("\n⚙️ --- KONFIGURASI ---")
        input_depth = input("⌨️  Masukkan Max Depth (Enter = Unlimited): ")
        if input_depth.strip() == "":
            MAX_DEPTH = None
            print("   -> Mode: Unlimited (None)")
        else:
            try:
                MAX_DEPTH = int(input_depth)
                print(f"   -> Max Depth: {MAX_DEPTH}")
            except:
                MAX_DEPTH = None
                print("   ⚠️ Input invalid. Default: Unlimited")

        # --- PREPARE DATA ---
        print(f"\n📂 Loading Data...")
        df_train = pd.read_csv(train_path)
        if 'id' in df_train.columns: df_train = df_train.drop('id', axis=1)
        
        X = df_train.drop('Target', axis=1).values
        y = df_train['Target'].values
        feature_names = df_train.drop('Target', axis=1).columns.tolist()

        # ==================================================================
        # 🧪 SKENARIO 1: VALIDASI INTERNAL
        # ==================================================================
        print("\n" + "="*50)
        print("🧪 SKENARIO 1: VALIDASI INTERNAL (Split Train/Val)")
        print("="*50)
        
        X_part_train, X_part_val, y_part_train, y_part_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print("🛠️  Melatih Model Scratch (80% Data)...")
        model_val = ID3DecisionTree(max_depth=MAX_DEPTH)
        model_val.fit(X_part_train, y_part_train, feature_names=feature_names)
        acc_scratch = accuracy_score(y_part_val, model_val.predict(X_part_val))

        print("🤖 Melatih Model Scikit-Learn (80% Data)...")
        sklearn_val = DecisionTreeClassifier(criterion='entropy', max_depth=MAX_DEPTH, random_state=42)
        sklearn_val.fit(X_part_train, y_part_train)
        acc_sklearn = accuracy_score(y_part_val, sklearn_val.predict(X_part_val))

        print(f"\n📊 HASIL RONDE 1 (AKURASI):")
        print(f"   1. ID3 From Scratch : {acc_scratch * 100:.2f}%")
        print(f"   2. Scikit-Learn     : {acc_sklearn * 100:.2f}%")

        # ==================================================================
        # 🚀 SKENARIO 2: FULL TRAINING & UJIAN
        # ==================================================================
        print("\n" + "="*50)
        print("🚀 SKENARIO 2: FULL TRAINING & UJIAN (Test.csv)")
        print("="*50)

        if os.path.exists(test_path):
            print("💪 Retraining Model dengan 100% Data Latih...")
            final_model = ID3DecisionTree(max_depth=MAX_DEPTH)
            final_model.fit(X, y, feature_names=feature_names)
            
            # Simpan Model Akhir (.pkl)
            final_model.save_model(save_model_path)

            # --- VISUALISASI ---
            print("\n🖼️  --- VISUALISASI (Model Full) ---")
            default_vis = MAX_DEPTH
            display_text = "Unlimited" if default_vis is None else str(default_vis)
            
            vis_input = input(f"⌨️  Depth Visualisasi (Enter = {display_text}): ")
            
            if vis_input.strip() == "":
                vis_depth = default_vis 
            else:
                try:
                    vis_depth = int(vis_input)
                except:
                    vis_depth = default_vis

            print("-" * 40)
            # Print ke Layar (Warna)
            final_model.print_tree(max_depth=vis_depth)
            print("-" * 40)
            
            # --- [BAGIAN BARU] SIMPAN TREE KE TXT ---
            try:
                with open(save_txt_path, 'w', encoding='utf-8') as f:
                    # Print ke File (Tanpa Warna)
                    final_model.print_tree(max_depth=vis_depth, file=f)
                print(f"📄 Struktur Tree (Text) disimpan ke: {save_txt_path}")
            except Exception as e:
                print(f"❌ Gagal menyimpan TXT: {e}")

            # PREDIKSI DATA TEST
            df_test = pd.read_csv(test_path)
            ids = df_test['id'] if 'id' in df_test.columns else range(len(df_test))
            if 'id' in df_test.columns: df_test = df_test.drop('id', axis=1)
            X_test_real = df_test.values

            print("\n🔮 Memprediksi data test.csv (Tanpa Kunci Jawaban)...")
            pred_scratch_test = final_model.predict(X_test_real)

            sub_scratch = pd.DataFrame({'id': ids, 'Target': pred_scratch_test})
            sub_scratch.to_csv(sub_scratch_path, index=False)

            print(f"✅ File Submission Siap: {sub_scratch_path}")
            print("   (Upload ke Kaggle untuk melihat Akurasi Test)")
        else:
            print("⚠️ File test.csv tidak ditemukan.")

    except Exception as e:
        print(f"❌ Terjadi Error: {e}")