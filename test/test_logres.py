import sys
from pathlib import Path
# Add project root to sys.path so we can import src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src.logistic_regression import SoftmaxRegression
from src.utils import one_hot_encode 


# 1. Create a larger synthetic dataset (2000 samples total, string labels)
rng = np.random.RandomState(0)
N_TOTAL = 2000
N1 = N_TOTAL // 3
N2 = N_TOTAL // 3
N3 = N_TOTAL - (N1 + N2)

class0 = rng.normal(loc=[1.0, 2.0], scale=0.3, size=(N1, 2))   # class A cluster
class1 = rng.normal(loc=[6.0, 9.0], scale=0.4, size=(N2, 2))   # class B cluster
class2 = rng.normal(loc=[1.0, 0.5], scale=0.25, size=(N3, 2))  # class C cluster

X_train = np.vstack([class0, class1, class2])

# Use string labels to demonstrate label encoding
y_str = np.array(["class_A"] * N1 + ["class_B"] * N2 + ["class_C"] * N3)

# Shuffle dataset
perm = rng.permutation(N_TOTAL)
X_train = X_train[perm]
y_str = y_str[perm]

# Label-encode string labels to integer labels 0..K-1
unique_labels, y_train = np.unique(y_str, return_inverse=True)

# One-hot encode if needed for external checks
y_one_hot = one_hot_encode(y_train)

# 2. Initialize model
model = SoftmaxRegression(
    lr=0.1,
    max_iter=100,
    n_classes=3,   # required for the current version of your implementation
    verbose=True   # show loss every few epochs
)

# 3. Train model
model.fit(X_train, y_one_hot)

print("\nFinal weights:")
print(model.weights)

print("\nLoss history (first 10):")
print(model.loss_history[:10])
print("Last loss:", model.loss_history[-1])

# 4. Test prediction on the training data itself
probs = model.predict_proba(X_train)
y_pred = model.predict(X_train)

print("\nPredicted probabilities:")
print(probs)

print("\nTrue labels:    ", y_train)
print("Predicted labels:", y_pred)

# 5. Check if the model looks reasonable:
#    ideally many predictions match y_train
correct = np.sum(y_pred == y_train)
print(f"\nCorrect predictions: {correct} out of {len(y_train)}")