from sklearn.linear_model import LogisticRegression
import numpy as np

class MLFilter:
    """
    Machine Learning filter for dynamic trade signal validation.
    Uses logistic regression on basic price action features.

    Attributes:
        model (LogisticRegression): Trained logistic regression model.
    """

    def __init__(self):
        self.model = LogisticRegression()

    def train(self, X_train, y_train):
        """
        Train the ML filter.

        Args:
            X_train (np.ndarray): Feature matrix.
            y_train (np.ndarray): Target labels.
        """
        self.model.fit(X_train, y_train)
        print("✅ ML filter trained successfully!")

    def predict(self, X):
        """
        Predict using the ML filter.

        Args:
            X (np.ndarray): Feature matrix to predict on.

        Returns:
            np.ndarray: Predictions (1 for allow trade, 0 for block trade).
        """
        return self.model.predict(X)
