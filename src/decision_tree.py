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

class ID3DecisionTree(BaseClassifier):
    def __init__(self, min_samples_split=2, max_depth=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None
        self.feature_types = []
        self.feature_names = []

    # --- TRAINING ---
    def fit(self, X, y, feature_names=None):

        if isinstance(X, pd.DataFrame):
            if feature_names is None: feature_names = X.columns.tolist()
            X_df = X.copy()
        else:
            X_df = pd.DataFrame(X)
        
        y = np.array(y)
        
        for col in X_df.columns:
            if X_df[col].isnull().any():
                if pd.api.types.is_numeric_dtype(X_df[col]): X_df[col] = X_df[col].fillna(X_df[col].mean())
                else: X_df[col] = X_df[col].fillna(X_df[col].mode()[0])
        
        X_vals = X_df.values

        self.feature_types = []
        n_features = X_vals.shape[1]
        for i in range(n_features):
            val = X_vals[0, i]
            self.feature_types.append('continuous' if isinstance(val, (int, float, np.number)) and not isinstance(val, str) else 'categorical')

        self.feature_names = feature_names if feature_names else [f"feat_{i}" for i in range(n_features)]
        
        self.root = self._id3(X_vals, y, list(range(n_features)), depth=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X, dtype=object)
        return np.array([self._traverse(x, self.root) for x in X])

    def _id3(self, X, y, attribute_indices, depth):
        n_samples = X.shape[0]
        n_labels = len(np.unique(y))

        if n_labels == 1: return Node(prediction=y[0])
        hit_max = (self.max_depth is not None and depth >= self.max_depth)
        if len(attribute_indices) == 0 or hit_max or n_samples < self.min_samples_split:
            return Node(prediction=self._most_common_label(y))

        best_feat, best_gain, best_thr = self._get_best_attribute(X, y, attribute_indices)
        if best_feat is None: return Node(prediction=self._most_common_label(y))

        node = Node()
        node.feature_idx = best_feat
        node.feature_name = self.feature_names[best_feat]
        node.is_continuous = (self.feature_types[best_feat] == 'continuous')

        if node.is_continuous:
            node.threshold = best_thr
            left_mask = X[:, best_feat] <= best_thr
            X_l, y_l = X[left_mask], y[left_mask]
            X_r, y_r = X[~left_mask], y[~left_mask]
            
            if len(X_l) == 0 or len(X_r) == 0: return Node(prediction=self._most_common_label(y))
            node.children['<='] = self._id3(X_l, y_l, attribute_indices, depth + 1)
            node.children['>'] = self._id3(X_r, y_r, attribute_indices, depth + 1)
        else:
            new_attrs = [a for a in attribute_indices if a != best_feat]
            for val in np.unique(X[:, best_feat]):
                mask = X[:, best_feat] == val
                X_sub, y_sub = X[mask], y[mask]
                child = Node(prediction=self._most_common_label(y)) if len(X_sub) == 0 else self._id3(X_sub, y_sub, new_attrs, depth + 1)
                node.children[val] = child
        return node

    def _get_best_attribute(self, X, y, attribute_indices):
        best_gain, best_feat, best_thr = -1, None, None
        for idx in attribute_indices:
            X_col = X[:, idx]
            if self.feature_types[idx] == 'continuous':
                thresholds = np.unique(X_col)
                if len(thresholds) > 50: thresholds = np.percentile(thresholds, np.linspace(0, 100, 10))
                for thr in thresholds:
                    gain = self._calc_gain_cont(y, X_col, thr)
                    if gain > best_gain: best_gain, best_feat, best_thr = gain, idx, thr
            else:
                gain = self._calc_gain_cat(y, X_col)
                if gain > best_gain: best_gain, best_feat, best_thr = gain, idx, None
        return best_feat, best_gain, best_thr

    def _calc_gain_cat(self, y, X_col):
        parent_ent = self._entropy(y)
        n, child_ent_sum = len(y), 0
        vals, counts = np.unique(X_col, return_counts=True)
        for val, count in zip(vals, counts):
            child_ent_sum += (count / n) * self._entropy(y[X_col == val])
        return parent_ent - child_ent_sum

    def _calc_gain_cont(self, y, X_col, thr):
        parent_ent = self._entropy(y)
        n = len(y)
        left = X_col <= thr
        if np.sum(left) == 0 or np.sum(~left) == 0: return 0
        y_l, y_r = y[left], y[~left]
        return parent_ent - ((len(y_l)/n)*self._entropy(y_l) + (len(y_r)/n)*self._entropy(y_r))

    def _entropy(self, y):
        ps = (np.bincount(y) if np.issubdtype(y.dtype, np.integer) else np.unique(y, return_counts=True)[1]) / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def _most_common_label(self, y):
        return None if len(y) == 0 else Counter(y).most_common(1)[0][0]

    def _traverse(self, x, node):
        if node.prediction is not None: return node.prediction
        val = x[node.feature_idx]
        if node.is_continuous:
            next_node = node.children.get('<=' if val <= node.threshold else '>')
        else:
            next_node = node.children.get(val)
        if next_node is None and node.children: next_node = list(node.children.values())[0]
        return self._traverse(x, next_node) if next_node else node.prediction

    def print_tree(self, node=None, prefix="", is_last=True, is_root=True, branch="", max_depth=None, cur_depth=0, file=None):
        if node is None: node = self.root
        if max_depth is not None and cur_depth > max_depth:
            if is_last and not is_root: print(f"{prefix}└── ...", file=file)
            return

        txt = f"Output: {node.prediction}" if node.prediction is not None else f"[{node.feature_name}]"
        if is_root: print(txt, file=file); new_prefix = ""
        else:
            branch_txt = f"({branch})"
            print(f"{prefix}{'└── ' if is_last else '├── '}{branch_txt} {txt}", file=file)
            new_prefix = prefix + ("    " if is_last else "│   ")

        if node.prediction is None:
            keys = list(node.children.keys())
            for i, k in enumerate(keys):
                self.print_tree(node.children[k], new_prefix, i == len(keys)-1, False, str(k), max_depth, cur_depth + 1, file)
