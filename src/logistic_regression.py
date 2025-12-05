import numpy as np
from src.core.base_model import BaseClassifier

class SoftmaxRegression(BaseClassifier):
    """
    Multinomial logistic regression (softmax regression) classifier.
    
    This model:
    - Learns a separate weight vector for each output class.
    - Uses the softmax function to produce a probability distribution
      over all classes for each input sample.
    - Is trained using gradient-based optimization on the categorical
      cross-entropy loss.
    """
    
    def __init__(self, lr=0.1, max_iter=100, n_classes=None, verbose=False, random_state=42):
        # Hyperparameters
        self.lr = lr
        self.max_iter = max_iter
        self.n_classes = n_classes
        self.verbose = verbose
        self.random_state = random_state
        # Create a per-instance random generator (deterministic when random_state is an int)
        try:
            self.rng = np.random.default_rng(self.random_state)
        except Exception:
            self.rng = np.random.default_rng(42)

        # Learned during training
        self.weights = None  # shape (n_features, n_features_with_bias)
        self.mean = None
        self.std = None
        self.loss_history = []
        self.classes_ = None
    
    def _add_bias(self, X: np.ndarray) -> np.ndarray:
        """
        Add a bias feature (column of ones) to the input data X.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Input feature matrix without bias.

        Returns
        -------
        X_bias : np.ndarray, shape (n_samples, n_features + 1)
            Input matrix with an additional bias column at the front.
        """

        bias = np.full((X.shape[0], 1), 1) # Concatenate a column of ones
        return np.hstack((bias, X)) # return X with bias column added
    
    def _softmax(self, Z: np.ndarray) -> np.ndarray:
        """
        Compute the softmax probabilities for each class.
        
        Parameters
        ----------
        Z : np.ndarray, shape (n_samples, n_classes)
            The linear scores for each class.

        Returns
        -------
        probs : np.ndarray, shape (n_samples, n_classes)
            The softmax probabilities for each class.
        
        Notes
        -----
        The softmax function is defined as:
            softmax(z_i) = exp(z_i) / sum_j exp(z_j)
        
        To avoid overflow issues, it is common to subtract the maximum
        score from each score before exponentiating.
        """
        # Subtract max for numerical stability, calculate exp scores
        exp_scores = np.exp(Z - np.max(Z, axis=1, keepdims=True))

        # Normalize to get probabilities
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    def _compute_loss(self, Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
        """
        Compute the categorical cross-entropy loss.
        
        Parameters
        ----------
        Y_true : np.ndarray, shape (n_samples, n_classes)
            One-hot encoded true class labels.
        Y_pred : np.ndarray, shape (n_samples, n_classes)
            Predicted class probabilities.

        Returns
        -------
        loss : float
            The average cross-entropy loss over all samples.

        Notes
        -----
        - The formula is: L = -1/N * sum_i sum_k Y_true[i, k] * log(P[i, k])
        - Only the probability of the true class contributes to the loss
          for each sample because Y_true is one-hot.
        """
        n_samples = Y_true.shape[0]
        
        # Calculate the log probabilities
        log_probs = np.log(Y_pred + 1e-15)  # Add small constant to avoid log(0)
    
        # Compute the cross-entropy loss
        loss = -np.sum(Y_true * log_probs) / n_samples
        return loss

    def fit(self, X: np.ndarray, y: np.ndarray, random_weights: bool=True):
        """
        Train the softmax regression model using gradient descent.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Input feature matrix.
        y : np.ndarray, shape (n_samples,)
            Class labels as integers.

        Steps
        -----
        1. Optionally standardize features and store mean and std.
        2. Add bias term to X.
        3. Initialize weight matrix W.
        4. For each epoch:
            a. Compute linear scores Z = X @ W.T
            b. Compute probabilities P = softmax(Z)
            c. Compute error (Y_true - P)
            d. Update W using gradient ascent with learning rate.
            e. Optionally compute and store loss for monitoring.

        Notes
        -----
        - y must be label encoded to integers 0..K-1 before calling fit.
        """
        # Determine number of classes
        if self.n_classes is None:
            if y.ndim == 2:
                self.n_classes = y.shape[1]
            else:
                self.classes_ = np.unique(y)
                self.n_classes = len(self.classes_)

        # Add bias term to input features
        X_bias = self._add_bias(X)

        # Initialize weights with reproducible randomness
        if random_weights:
            # Random small weights seeded by self.rng
            self.weights = self.rng.standard_normal((self.n_classes, X_bias.shape[1])) * 0.01
        else:
            # Zero initialization
            self.weights = np.zeros((self.n_classes, X_bias.shape[1]))

        if self.verbose:
            print("Starting training...")
            for epoch in range(self.max_iter):
                # Compute linear scores
                Z = X_bias @ self.weights.T  # shape (n_samples, n_classes)

                # Compute softmax probabilities
                probs = self._softmax(Z)  # shape (n_samples, n_classes)

                # Compute the error
                error = y - probs  # shape (n_samples, n_classes)

                # Compute gradient and update weights
                gradient = (error.T @ X_bias) / X_bias.shape[0]  # shape (n_classes, n_features + 1)
                self.weights += self.lr * gradient  # Update weights

                # Compute loss for monitoring
                loss = self._compute_loss(y, probs)
                self.loss_history.append(loss) # Store loss in history

                if epoch % 10 == 0 or epoch == self.max_iter - 1:
                    print(f"Epoch {epoch+1}/{self.max_iter}, Loss: {loss:.4f}")
        else:
            # Training loop
            for epoch in range(self.max_iter):
                # Compute linear scores
                Z = X_bias @ self.weights.T  # shape (n_samples, n_classes)

                # Compute softmax probabilities
                probs = self._softmax(Z)  # shape (n_samples, n_classes)

                # Compute the error
                error = y - probs  # shape (n_samples, n_classes)

                # Compute gradient and update weights
                gradient = (error.T @ X_bias) / X_bias.shape[0]  # shape (n_classes, n_features + 1)
                self.weights += self.lr * gradient  # Update weights

                # Compute loss for monitoring
                loss = self._compute_loss(y, probs)
                self.loss_history.append(loss) # Store loss in history
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for input samples.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Input feature matrix.

        Returns
        -------
        probs : np.ndarray, shape (n_samples, n_classes)
            Predicted class probabilities for each sample.
        """

        # Calculate Z score for each label
        W = self.weights  # shape (n_classes, n_features + 1)
        X_bias = self._add_bias(X) # add bias term

        Z = X_bias @ W.T  # calculate linear scores
        return self._softmax(Z)  # return softmax probabilities


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for input samples.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Input feature matrix.

        Returns
        -------
        y_pred : np.ndarray, shape (n_samples,)
            Predicted class labels as integers.
        """
        probs = self.predict_proba(X)  # shape (n_samples, n_classes)
        return np.argmax(probs, axis=1)  # return class with highest probability