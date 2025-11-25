import sys
from pathlib import Path

# Add project root to sys.path so we can import src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src.logistic_regression import SoftmaxRegression


# 1. Create a small dummy dataset
# e.g., 6 samples, 2 features, 3 classes (0, 1, 2)
X_train = np.array([
    [1.0, 2.0],   # likely class 0
    [1.5, 1.8],   # likely class 0
    [5.0, 8.0],   # likely class 1
    [6.0, 9.0],   # likely class 1
    [1.0, 0.5],   # likely class 2
    [0.5, 1.0],   # likely class 2
])

y_train = np.array([0, 0, 1, 1, 2, 2])  # labels already 0..2

# 2. Initialize model
model = SoftmaxRegression(
    lr=0.1,
    max_iter=100,
    n_classes=3,   # required for the current version of your implementation
    verbose=True   # show loss every few epochs
)

# 3. Train model
model.fit(X_train, y_train)

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