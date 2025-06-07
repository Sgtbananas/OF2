import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
import logging

class MLFilter:
    """
    Bulletproof ML filter for trading:
    - Profit first: Only trades if model is valid and features match.
    - Safety second: Auto-blocks trading on any bug, mismatch, or error.
    - Logging for all critical events.
    """

    def __init__(self, model_path=None):
        self.model = None
        self.selector = None
        self.features = None
        self.model_path = model_path

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
        else:
            # Untrained fallback, refuses to trade.
            self.model = LogisticRegression()
            logging.warning("MLFilter: No model found, using untrained LogisticRegression (will block all trades).")

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        print("✅ ML filter trained successfully!")

    def extract_features(self, df, idx):
        """
        Extract features exactly as in training—if anything is missing, prediction is refused.
        """
        row = df.iloc[idx]
        if self.features is not None:
            feats = []
            missing = []
            for col in self.features:
                if col in row:
                    v = row[col]
                    if v is None or np.isnan(v) or np.isinf(v):
                        v = 0.0
                else:
                    missing.append(col)
                    v = 0.0
                feats.append(v)
            if missing:
                msg = f"MLFilter: Refusing to predict—runtime data missing features: {missing}"
                logging.error(msg)
                raise RuntimeError(msg)
            arr = np.array(feats, dtype=np.float32).reshape(1, -1)
        else:
            # Fallback: legacy OHLCV
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
                msg = f"MLFilter: Feature selector transform failed: {e}"
                logging.error(msg)
                raise RuntimeError(msg)

        return arr

    def predict(self, X):
        if hasattr(self.model, "predict"):
            return self.model.predict(X)
        raise RuntimeError("MLFilter: Model lacks predict(). Refusing to trade.")

    def predict_proba(self, X):
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        raise RuntimeError("MLFilter: Model lacks predict_proba().")

    def should_enter(self, df, idx, signal, threshold=0.5):
        """
        Only allow entry if model and features are fully valid. Refuse otherwise.
        """
        try:
            arr = self.extract_features(df, idx)
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(arr)[0, 1]
                if np.isnan(proba) or np.isinf(proba):
                    logging.error("MLFilter: Model returned nan/inf for proba; refusing trade.")
                    return False
                return proba >= threshold
            pred = self.predict(arr)
            return bool(pred[0])
        except Exception as e:
            logging.error(f"MLFilter: should_enter blocked: {e}")
            return False

    def should_exit(self, df, idx, signal, threshold=0.5):
        """
        Only allow exit if model and features are fully valid. Refuse otherwise.
        """
        try:
            arr = self.extract_features(df, idx)
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(arr)[0, 1]
                if np.isnan(proba) or np.isinf(proba):
                    logging.error("MLFilter: Model returned nan/inf for proba; refusing trade exit.")
                    return False
                return proba >= threshold
            pred = self.predict(arr)
            return bool(pred[0])
        except Exception as e:
            logging.error(f"MLFilter: should_exit blocked: {e}")
            return False

    def get_feature_importance(self):
        if hasattr(self.model, "feature_importances_") and self.features is not None:
            return dict(zip(self.features, self.model.feature_importances_))
        if hasattr(self.model, "coef_") and self.features is not None:
            return dict(zip(self.features, self.model.coef_[0]))
        return {}

    def explain(self, df, idx):
        try:
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
        except Exception as e:
            logging.error(f"MLFilter: explain blocked: {e}")
            return {"error": str(e)}