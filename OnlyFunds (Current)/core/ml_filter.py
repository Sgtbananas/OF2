import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib
import os

class MLFilter:
    """
    Machine Learning filter for dynamic trade signal validation.
    Uses logistic regression on basic price action features.
    """
    def __init__(self, model_path="ml_filter_model.pkl"):
        self.model_path = model_path
        self.model = LogisticRegression()
        self.is_trained = False
        # Try to load pre-trained model
        if os.path.exists(self.model_path):
            self.load(self.model_path)

    def train(self, X_train, y_train):
        """
        Train the ML filter and persist model.
        """
        self.model.fit(X_train, y_train)
        self.is_trained = True
        self.save(self.model_path)
        print("✅ ML filter trained and saved successfully!")

    def predict(self, X):
        """
        Predict using the ML filter.
        Returns: np.array of predictions (1 for allow trade, 0 for block trade).
        """
        if not self.is_trained:
            # Allow all trades if not trained
            return np.ones((np.array(X).shape[0],), dtype=int)
        return self.model.predict(X)

    def save(self, path=None):
        """Save model to disk."""
        if not path:
            path = self.model_path
        joblib.dump(self.model, path)

    def load(self, path=None):
        """Load model from disk."""
        if not path:
            path = self.model_path
        try:
            self.model = joblib.load(path)
            self.is_trained = True
            print("✅ ML filter loaded from disk.")
        except Exception as e:
            print(f"⚠️ Could not load model: {e}; will need training.")

    @staticmethod
    def extract_features(df, i):
        """
        Example feature extraction (expand as needed):
        - 5-period rolling stddev of close returns
        - 1-period close diff
        """
        return [
            df["close"].pct_change().rolling(5).std().iloc[i] or 0,
            df["close"].diff().iloc[i] or 0
        ]