import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib
import os

class MLFilter:
    """
    Machine Learning filter for dynamic trade signal validation.
    Uses logistic regression on basic price action features.
    """
    def __init__(self, model_path: str = "ml_filter_model.pkl"):
        self.model_path = model_path
        self.model = LogisticRegression()
        self.is_trained = False
        # Try to load pre-trained model
        if os.path.exists(self.model_path):
            self.load(self.model_path)

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Train the ML filter and persist model.
        """
        if len(X_train) == 0 or len(y_train) == 0:
            raise ValueError("Training data is empty.")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        self.save(self.model_path)
        print("✅ ML filter trained and saved successfully!")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the ML filter.
        Returns: np.array of predictions (1 for allow trade, 0 for block trade).
        """
        if not self.is_trained:
            # Allow all trades if not trained
            return np.ones((np.array(X).shape[0],), dtype=int)
        return self.model.predict(X)

    def save(self, path: str = None):
        """Save model to disk."""
        if not path:
            path = self.model_path
        try:
            joblib.dump(self.model, path)
        except Exception as e:
            print(f"⚠️ Could not save model: {e}")

    def load(self, path: str = None):
        """Load model from disk."""
        if not path:
            path = self.model_path
        try:
            self.model = joblib.load(path)
            self.is_trained = True
            print("✅ ML filter loaded from disk.")
        except Exception as e:
            print(f"⚠️ Could not load model: {e}; will need training.")
            self.is_trained = False

    @staticmethod
    def extract_features(df, i):
        """
        Example feature extraction (expand as needed):
        - 5-period rolling stddev of close returns
        - 1-period close diff

        Args:
            df (pd.DataFrame): DataFrame with price data. Must include 'close' or 'Close' column.
            i (int): Index for extraction.

        Returns:
            list: Feature vector for row i.
        """
        # Support both 'close' and 'Close'
        col = 'close' if 'close' in df.columns else 'Close'
        return [
            df[col].pct_change().rolling(5).std().iloc[i] or 0,
            df[col].diff().iloc[i] or 0
        ]

if __name__ == "__main__":
    # Basic CLI/test usage example
    import pandas as pd
    # Generate dummy data
    df = pd.DataFrame({"Close": np.linspace(100, 110, 50) + np.random.randn(50)})
    X = [MLFilter.extract_features(df, i) for i in range(5, len(df))]
    y = np.random.randint(0, 2, len(X))
    ml = MLFilter()
    ml.train(np.array(X), y)
    print("Sample predictions:", ml.predict(np.array(X)))