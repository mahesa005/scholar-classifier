import numpy as np
from cvxopt import matrix, solvers
import numpy as np
from src.core.base_model import BaseClassifier

class SVM_QP(BaseClassifier):
    def __init__(self, C=1.0, gamma=0.1, kernel='rbf'):
        self.C = C
        self.gamma = gamma
        self.kernel = kernel
        self.alphas = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.b = 0

    def linear_kernel(self, X1, X2):
        return np.dot(X1, X2.T)

    def rbf_kernel(self, X1, X2):
        # Menggunakan Euclidean distance squared
        n_samples_1 = X1.shape[0]
        n_samples_2 = X2.shape[0]
        K = np.zeros((n_samples_1, n_samples_2))
        
        for i in range(n_samples_1):
            for j in range(n_samples_2):
                diff = X1[i] - X2[j]
                K[i, j] = np.exp(-self.gamma * np.dot(diff, diff))
        return K

    def fit(self, X, y):

        # Konversi ke numpy array jika dia DataFrame
        if hasattr(X, 'values'):
            X = X.values 
        
        if hasattr(y, 'values'):
            y = y.values

        n_samples, n_features = X.shape
        y = y.astype(float).reshape(-1, 1) 
        
        # Matriks Gram (Kernel Matrix)
        if self.kernel == 'linear':
            K = self.linear_kernel(X, X)
        elif self.kernel == 'rbf':
            K = self.rbf_kernel(X, X)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
        
        # Matriks P, q, G, h, A, b untuk CVXOPT
        # P = outer_product(y) * K
        P = matrix(np.outer(y, y) * K)
        
        # q = vektor -1
        q = matrix(-np.ones((n_samples, 1)))
        
        # G dan h untuk constraint: 0 <= alpha <= C
        # -alpha <= 0  --> G_std = -I, h_std = 0
        # alpha <= C   --> G_slack = I, h_slack = C
        G_std = -np.eye(n_samples)
        G_slack = np.eye(n_samples)
        G = matrix(np.vstack((G_std, G_slack)))
        
        h_std = np.zeros(n_samples)
        h_slack = np.ones(n_samples) * self.C
        h = matrix(np.hstack((h_std, h_slack)))
        
        # A dan b untuk constraint: sum(alpha * y) = 0
        A = matrix(y.T)
        b = matrix(np.zeros(1))

        # QP menggunakan Solver
        solvers.options['show_progress'] = False 
        solution = solvers.qp(P, q, G, h, A, b)
        
        # Ambil nilai alpha (Lagrange multipliers)
        alphas = np.ravel(solution['x'])
        
        # Filter Support Vectors (alpha > threshold kecil mendekati 0)
        idx = alphas > 1e-5
        self.alphas = alphas[idx]
        self.support_vectors = X[idx]
        self.support_vector_labels = y[idx]
        
        # Hitung Bias (b)
        # b = mean(y_k - sum(alpha_i * y_i * K(x_i, x_k))) untuk semua support vector k
        bias_list = []
        for i in range(len(self.alphas)):
            pred_val = 0
            for j in range(len(self.alphas)):
                if self.kernel == 'linear':
                    k_val = np.dot(self.support_vectors[j], self.support_vectors[i])
                else:  # rbf
                    k_val = np.exp(-self.gamma * np.linalg.norm(self.support_vectors[j] - self.support_vectors[i])**2)
                pred_val += self.alphas[j] * self.support_vector_labels[j] * k_val
            
            bias_list.append(self.support_vector_labels[i] - pred_val)
            
        # rata-rata bias
        self.b = np.mean(bias_list) if bias_list else 0

    def predict(self, X):
        y_pred = []
        # Prediksi: sign(sum(alpha_i * y_i * K(x_i, x_new)) + b)
        for x in X:
            prediction = 0
            for i in range(len(self.alphas)):
                # Kernel antara data baru (x) dan support vector (sv)
                if self.kernel == 'linear':
                    k_val = np.dot(self.support_vectors[i], x)
                else:  # rbf
                    k_val = np.exp(-self.gamma * np.linalg.norm(self.support_vectors[i] - x)**2)
                prediction += self.alphas[i] * self.support_vector_labels[i] * k_val
            
            prediction += self.b
            y_pred.append(np.sign(prediction))
            
        return np.array(y_pred)

class DAGSVM(BaseClassifier):
    def __init__ (self, C=1.0, gamma=0.1, kernel='rbf'):
        self.C = C
        self.gamma = gamma
        self.kernel = kernel
        self.classifiers = {}
        self.classes = None
    
    def fit(self, X, y):
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        # Melatih SVM untuk setiap pasangan kelas
        for i in range(n_classes):
            for j in range(i + 1, n_classes):
                class_i = self.classes[i]
                class_j = self.classes[j]
                
                # Filter data untuk kelas i dan j
                idx = np.where((y == class_i) | (y == class_j))
                X_ij = X[idx]
                y_ij = y[idx]
                
                # Ubah label menjadi +1 dan -1
                y_ij = np.where(y_ij == class_i, 1, -1)
                
                # Latih SVM
                svm_ij = SVM_QP(C=self.C, gamma=self.gamma, kernel=self.kernel)
                svm_ij.fit(X_ij, y_ij)
                
                # Simpan classifier
                self.classifiers[(class_i, class_j)] = svm_ij

    def predict(self, X):
        y_pred = []
        
        for x in X:
            candidates = list(self.classes)

            while len(candidates) > 1:
                # Ambil kandidat Paling Kiri (Awal) dan Paling Kanan (Akhir)
                c_first = candidates[0]
                c_last = candidates[-1]
                
                if (c_first, c_last) in self.classifiers:
                    model = self.classifiers[(c_first, c_last)]
                    pred = model.predict(x.reshape(1, -1))
                    
                    if pred == 1:
                        candidates.pop(-1)
                    else:
                        candidates.pop(0)
                        
                elif (c_last, c_first) in self.classifiers:
                    model = self.classifiers[(c_last, c_first)]
                    pred = model.predict(x.reshape(1, -1))
                    
                    if pred == 1:
                        candidates.pop(0)
                    else:
                        candidates.pop(-1)
                else:
                    raise ValueError(f"Model untuk pasangan {c_first}-{c_last} tidak ditemukan!")
            
            y_pred.append(candidates[0])

        return np.array(y_pred)
        