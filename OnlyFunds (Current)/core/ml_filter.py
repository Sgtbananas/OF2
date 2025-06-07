import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
import logging

class MLFilter:
    """
    World-class Machine Learning filter for algorithmic trading signals.
    Handles robust model, selector, and feature name loading for both legacy and modern pipelines.

    Attributes:
        model: Trained sklearn-compatible model (e.g., RandomForest, LogisticRegression, etc.)
        selector: Feature selector (e.g., from SelectFromModel) or None
        features: List of feature names used for training (or None)
        model_path: Path to loaded model file
        model_version: (Optional) Model class name/version for future compatibility
    """

    def __init__(self, model_path=None):
        self.model = None
        self.selector = None
        self.features = None
        self.model_path = model_path
        self.model_version = None

        # Load model and optionally selector/features
        if model_path and os.path.exists(model_path):
            loaded = joblib.load(model_path)
            if isinstance(loaded, tuple):
                self.model = loaded[0]
                if len(loaded) > 1:
                    self.selector = loaded[1]
                if len(loaded) > 2:
                    self.features = loaded[2]
                logging.info(f"MLFilter: Loaded model, selector, features from {model_path}")
            else:
                self.model = loaded
                logging.info(f"MLFilter: Loaded model only from {model_path}")
            self.model_version = type(self.model).__name__
        else:
            self.model = LogisticRegression()
            logging.warning("MLFilter: No model found, using new (untrained) LogisticRegression.")

    def train(self, X_train, y_train):
        """
        Train the ML filter model (for dev/test use).
        """
        self.model.fit(X_train, y_train)
        print("✅ ML filter trained successfully!")

    def extract_features(self, df, idx):
        """
        Extract features in the same order as training.
        If features were saved, use them. Otherwise, fallback to OHLCV.
        Args:
            df (pd.DataFrame): DataFrame with price data.
            idx (int): Index to extract features from.
        Returns:
            np.ndarray: Feature vector for the ML model.
        """
        row = df.iloc[idx]
        if self.features is not None:
            feats = []
            for col in self.features:
                v = row[col] if col in row else 0.0
                if v is None or np.isnan(v) or np.isinf(v):
                    v = 0.0
                feats.append(v)
            arr = np.array(feats, dtype=np.float32).reshape(1, -1)
        else:
            # fallback: standard OHLCV
            arr = np.array([
                row.get("close", 0.0),
                row.get("high", 0.0),
                row.get("low", 0.0),
                row.get("volume", 0.0)
            ], dtype=np.float32).reshape(1, -1)
        # Apply selector if present
        if self.selector is not None:
            try:
                arr = self.selector.transform(arr)
            except Exception as e:
                logging.error(f"MLFilter: Feature selector transform failed: {e}")
        return arr

    def predict(self, X):
        """
        Predict class (0/1) for input X.
        Args:
            X (np.ndarray): Feature matrix
        Returns:
            np.ndarray: Predictions (1=allow trade, 0=block trade)
        """
        if hasattr(self.model, "predict"):
            return self.model.predict(X)
        raise RuntimeError("MLFilter: Model lacks predict() method!")

    def predict_proba(self, X):
        """
        Optionally return probability of positive class (for advanced thresholding).
        """
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        raise RuntimeError("MLFilter: Model lacks predict_proba() method!")

    def should_enter(self, df, idx, signal, threshold=0.5):
        """
        Decide whether to allow entry based on ML model prediction.
        Uses threshold if model supports predict_proba.
        """
        arr = self.extract_features(df, idx)
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(arr)[0, 1]
            return proba >= threshold
        pred = self.predict(arr)
        return bool(pred[0])

    def should_exit(self, df, idx, signal, threshold=0.5):
        """
        Decide whether to allow exit based on ML model prediction.
        Uses threshold if model supports predict_proba.
        """
        arr = self.extract_features(df, idx)
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(arr)[0, 1]
            return proba >= threshold
        pred = self.predict(arr)
        return bool(pred[0])

    def get_feature_importance(self):
        """
        Return feature importance if available (for explainability).
        """
        if hasattr(self.model, "feature_importances_"):
            return dict(zip(self.features if self.features is not None else [], self.model.feature_importances_))
        if hasattr(self.model, "coef_"):
            return dict(zip(self.features if self.features is not None else [], self.model.coef_[0]))
        return {}

    def explain(self, df, idx):
        """
        Optionally: Return a dict with input features, prediction, and importance for explainability.
        """
        feats = self.extract_features(df, idx)
        pred = self.predict(feats)
        info = {
            "features": dict(zip(self.features if self.features is not None else [], feats.flatten())),
            "prediction": int(pred[0]),
            "importance": self.get_feature_importance()
        }
        if hasattr(self.model, "predict_proba"):
            info["probability"] = float(self.model.predict_proba(feats)[0, 1])
        return info