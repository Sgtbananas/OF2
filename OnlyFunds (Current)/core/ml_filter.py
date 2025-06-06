from sklearn.linear_model import LogisticRegression
import numpy as np
import joblib
import os

class MLFilter:
    """
    Machine Learning filter for dynamic trade signal validation.
    Uses logistic regression on basic price action features.

    Attributes:
        model (LogisticRegression): Trained logistic regression model.
    """

    def __init__(self, model_path=None):
        if model_path and os.path.exists(model_path):
            self.model = joblib.load(model_path)
        else:
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

    def extract_features(self, df, idx):
        """
        Extracts features from the DataFrame at the given index.
        You should extend this method with your real feature engineering.

        Args:
            df (pd.DataFrame): DataFrame with price data.
            idx (int): Index to extract features from.

        Returns:
            np.ndarray: Feature vector for the ML model.
        """
        row = df.iloc[idx]
        # Example: using close, high, low, volume
        features = [
            row.get("close", 0),
            row.get("high", 0),
            row.get("low", 0),
            row.get("volume", 0)
        ]
        # Add more sophisticated features as needed
        return np.array(features).reshape(1, -1)

    def should_enter(self, df, idx, signal):
        """
        Decide whether to allow entry based on ML model prediction.

        Args:
            df (pd.DataFrame): DataFrame with price data.
            idx (int): Index to extract features for.
            signal (int): Trading signal.

        Returns:
            bool: True if model allows entry, False otherwise.
        """
        features = self.extract_features(df, idx)
        # Optionally, you could use signal as a feature too.
        pred = self.predict(features)
        return bool(pred[0])

    def should_exit(self, df, idx, signal):
        """
        Decide whether to allow exit based on ML model prediction.

        Args:
            df (pd.DataFrame): DataFrame with price data.
            idx (int): Index to extract features for.
            signal (int): Trading signal.

        Returns:
            bool: True if model allows exit, False otherwise.
        """
        features = self.extract_features(df, idx)
        pred = self.predict(features)
        return bool(pred[0])