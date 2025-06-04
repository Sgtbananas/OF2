import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import joblib
import os

class MLFilter:
    """
    Advanced Machine Learning filter for dynamic trade signal validation.
    Supports both RandomForest and LogisticRegression.
    Feature set includes price action, technical indicators, and more.
    """
    def __init__(self, model_path: str = "ml_filter_model.pkl", model_type: str = "random_forest"):
        self.model_path = model_path
        self.model_type = model_type
        self.is_trained = False
        self.model = self._init_model(model_type)
        # Try to load pre-trained model
        if os.path.exists(self.model_path):
            self.load(self.model_path)

    def _init_model(self, model_type):
        if model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=200,
                max_depth=6,
                random_state=42,
                class_weight='balanced'
            )
        elif model_type == "logistic_regression":
            return LogisticRegression(
                solver='lbfgs',
                max_iter=1000,
                class_weight='balanced',
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def train(self, X_train: np.ndarray, y_train: np.ndarray, eval_split: float = 0.2, grid_search: bool = False):
        """
        Train the ML filter and persist the model. Optionally performs grid search.
        """
        if len(X_train) == 0 or len(y_train) == 0:
            raise ValueError("Training data is empty.")

        X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=eval_split, random_state=42)

        # Optionally run grid search
        if grid_search and self.model_type == "random_forest":
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [4, 6, 8, None]
            }
            grid = GridSearchCV(self.model, param_grid, scoring='f1', cv=3, n_jobs=-1)
            grid.fit(X_tr, y_tr)
            self.model = grid.best_estimator_
            print(f"✅ Grid search best params: {grid.best_params_}")
        else:
            self.model.fit(X_tr, y_tr)

        self.is_trained = True
        self.save(self.model_path)
        print("✅ ML filter trained and saved successfully!")

        # Validation report
        preds = self.model.predict(X_val)
        print("MLFilter validation report:\n", classification_report(y_val, preds, digits=3))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the ML filter.
        Returns: np.array of predictions (1 for allow trade, 0 for block trade).
        """
        if not self.is_trained:
            # Allow all trades if not trained
            return np.ones((np.array(X).shape[0],), dtype=int)
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return prediction probabilities (probability of allowing trade).
        """
        if not self.is_trained:
            return np.ones((np.array(X).shape[0],), dtype=float)
        return self.model.predict_proba(X)[:, 1]  # probability for class '1'

    def save(self, path: str = None):
        """Save model to disk."""
        if not path:
            path = self.model_path
        try:
            joblib.dump({
                "model": self.model,
                "model_type": self.model_type
            }, path)
        except Exception as e:
            print(f"⚠️ Could not save model: {e}")

    def load(self, path: str = None):
        """Load model from disk."""
        if not path:
            path = self.model_path
        try:
            bundle = joblib.load(path)
            self.model = bundle["model"]
            self.model_type = bundle.get("model_type", "random_forest")
            self.is_trained = True
            print("✅ ML filter loaded from disk.")
        except Exception as e:
            print(f"⚠️ Could not load model: {e}; will need training.")
            self.is_trained = False

    @staticmethod
    def extract_features(df: pd.DataFrame, i: int) -> list:
        """
        Robust feature extraction for ML trading filter.
        Features:
            - Rolling volatility (std)
            - Price change
            - Rolling mean
            - RSI
            - MACD diff
            - Volume z-score
            - Optional: add your own domain features

        Args:
            df (pd.DataFrame): DataFrame with OHLCV columns.
            i (int): Index to compute features for.

        Returns:
            list: Feature vector for row i.
        """
        # Defensive: support 'close' or 'Close', 'volume' or 'Volume'
        col_close = 'close' if 'close' in df.columns else 'Close'
        col_vol = 'volume' if 'volume' in df.columns else 'Volume'
        # Feature 1: 5-period rolling stddev of returns
        f1 = df[col_close].pct_change().rolling(5).std().iloc[i] if i >= 5 else 0
        # Feature 2: 1-period price change
        f2 = df[col_close].diff().iloc[i] if i > 0 else 0
        # Feature 3: 14-period rolling mean
        f3 = df[col_close].rolling(14).mean().iloc[i] if i >= 14 else 0
        # Feature 4: RSI
        f4 = MLFilter._rsi(df[col_close], i, period=14)
        # Feature 5: MACD diff (12-26 EMA)
        f5 = MLFilter._macd_diff(df[col_close], i)
        # Feature 6: Volume z-score
        f6 = MLFilter._zscore(df[col_vol], i, window=20) if col_vol in df.columns else 0
        return [f1, f2, f3, f4, f5, f6]

    @staticmethod
    def _rsi(series: pd.Series, i: int, period: int = 14) -> float:
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[i] if i >= period else 0

    @staticmethod
    def _macd_diff(series: pd.Series, i: int, fast: int = 12, slow: int = 26) -> float:
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        return macd.iloc[i] if i >= slow else 0

    @staticmethod
    def _zscore(series: pd.Series, i: int, window: int = 20) -> float:
        if i < window:
            return 0
        mean = series.rolling(window).mean().iloc[i]
        std = series.rolling(window).std().iloc[i]
        return (series.iloc[i] - mean) / (std + 1e-9) if std > 0 else 0

if __name__ == "__main__":
    # Example: Train and test the advanced MLFilter on dummy data
    import pandas as pd
    np.random.seed(42)
    # Simulate dummy OHLCV data
    n = 150
    df = pd.DataFrame({
        "Close": np.cumsum(np.random.randn(n)) + 100,
        "Volume": np.abs(np.random.randn(n)) * 1000
    })
    # Calculate features for each row (skip initial periods)
    X, y = [], []
    for i in range(30, len(df)):
        feats = MLFilter.extract_features(df, i)
        X.append(feats)
        # Example label: positive price change in next bar
        y.append(int(df["Close"].iloc[i+1] > df["Close"].iloc[i]) if i+1 < len(df) else 0)
    ml = MLFilter(model_type="random_forest")
    ml.train(np.array(X), np.array(y), grid_search=True)
    print("Sample predictions:", ml.predict(np.array(X[-5:])))
    print("Sample probabilities:", ml.predict_proba(np.array(X[-5:])))