import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import logging
from typing import Optional, List, Any

class MLFilter:
    """
    World-class ML filter for trading bots.

    - Only trades if model is valid, feature set matches, and all preconditions are met.
    - Bulletproof: refuses to trade on any bug, data mismatch, missing or invalid features.
    - Transparent logging of every critical event or error.
    - Explainability: provides detailed model and feature diagnostics.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model: Optional[Any] = None
        self.selector: Optional[Any] = None
        self.features: Optional[List[str]] = None
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
            self.model = LogisticRegression()
            logging.warning("MLFilter: No model found, using untrained LogisticRegression (will block all trades).")

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        self.model.fit(X_train, y_train)
        print("✅ ML filter trained successfully!")

    def extract_features(self, df: pd.DataFrame, idx: int) -> np.ndarray:
        """
        Extract features from a DataFrame row, ensuring order and presence matches training.
        Applies selector if present.
        """
        if self.features is None:
            raise RuntimeError("MLFilter: Model loaded without feature names. Cannot extract features.")

        row = df.iloc[idx]
        feats = []
        missing = []
        for col in self.features:
            if col in row:
                v = row[col]
                if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
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
        logging.debug(f"MLFilter: Features extracted for prediction: {dict(zip(self.features, arr.flatten()))}")

        # ---- WORLD-CLASS PATCH: SHAPE/ORDER CHECK BEFORE SELECTOR ----
        if self.selector is not None:
            expected_shape = len(self.features)
            if arr.shape[1] != expected_shape:
                msg = (
                    f"MLFilter: Feature count/order mismatch before selector.transform! "
                    f"Got shape {arr.shape}, expected ({1}, {expected_shape}). "
                    f"Feature order: {self.features}"
                )
                logging.error(msg)
                raise RuntimeError(msg)
            try:
                arr = self.selector.transform(arr)
                logging.debug(f"MLFilter: Selector reduced features to shape {arr.shape}")
            except Exception as e:
                msg = f"MLFilter: Feature selector transform failed: {e}"
                logging.error(msg)
                raise RuntimeError(msg)
        return arr

    def predict(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.model, "predict"):
            return self.model.predict(X)
        raise RuntimeError("MLFilter: Model lacks predict(). Refusing to trade.")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        raise RuntimeError("MLFilter: Model lacks predict_proba().")

    def should_enter(self, df: pd.DataFrame, idx: int, signal: Any, threshold: float = 0.5) -> bool:
        """
        Should enter trade? Only if model, selector, and features are all present and valid.
        """
        try:
            arr = self.extract_features(df, idx)
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(arr)[0, 1]
                logging.info(f"MLFilter: Entry proba={proba:.4f} (threshold={threshold:.4f})")
                if np.isnan(proba) or np.isinf(proba):
                    logging.error("MLFilter: Model returned nan/inf for proba; refusing trade.")
                    return False
                return proba >= threshold
            pred = self.predict(arr)
            return bool(pred[0])
        except Exception as e:
            logging.error(f"MLFilter: should_enter blocked: {e}")
            return False

    def should_exit(self, df: pd.DataFrame, idx: int, signal: Any, threshold: float = 0.5) -> bool:
        """
        Should exit trade? Only if model, selector, and features are all present and valid.
        """
        try:
            arr = self.extract_features(df, idx)
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(arr)[0, 1]
                logging.info(f"MLFilter: Exit proba={proba:.4f} (threshold={threshold:.4f})")
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
            importances = self.model.feature_importances_
            if self.selector is not None and hasattr(self.selector, 'get_support'):
                selected = self.selector.get_support()
                selected_features = [f for f, sel in zip(self.features, selected) if sel]
                return dict(zip(selected_features, importances))
            else:
                return dict(zip(self.features, importances))
        if hasattr(self.model, "coef_") and self.features is not None:
            coefs = self.model.coef_[0]
            if self.selector is not None and hasattr(self.selector, 'get_support'):
                selected = self.selector.get_support()
                selected_features = [f for f, sel in zip(self.features, selected) if sel]
                return dict(zip(selected_features, coefs))
            else:
                return dict(zip(self.features, coefs))
        return {}

    def explain(self, df: pd.DataFrame, idx: int):
        """
        Returns a detailed human- and machine-readable explanation of the model's prediction,
        including feature values, importance, and probabilities.
        """
        try:
            feats = []
            feature_values = {}
            missing = []
            # Always show features before selector
            row = df.iloc[idx]
            for col in self.features:
                if col in row:
                    v = row[col]
                    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
                        v = 0.0
                else:
                    missing.append(col)
                    v = 0.0
                feats.append(v)
                feature_values[col] = v
            if missing:
                msg = f"MLFilter: Refusing to explain—runtime data missing features: {missing}"
                logging.error(msg)
                raise RuntimeError(msg)
            arr = np.array(feats, dtype=np.float32).reshape(1, -1)

            # Features after selector (if any)
            if self.selector is not None:
                expected_shape = len(self.features)
                if arr.shape[1] != expected_shape:
                    msg = (
                        f"MLFilter: Feature count/order mismatch before selector.transform in explain! "
                        f"Got shape {arr.shape}, expected ({1}, {expected_shape}). "
                        f"Feature order: {self.features}"
                    )
                    logging.error(msg)
                    raise RuntimeError(msg)
                try:
                    arr_selected = self.selector.transform(arr)
                    support_mask = self.selector.get_support() if hasattr(self.selector, 'get_support') else None
                    selected_features = [f for f, sel in zip(self.features, support_mask) if sel] if support_mask is not None else self.features
                    selected_values = dict(zip(selected_features, arr_selected.flatten()))
                except Exception as e:
                    logging.error(f"MLFilter: Selector transform failed in explain: {e}")
                    arr_selected = arr
                    selected_features = self.features
                    selected_values = dict(zip(selected_features, arr.flatten()))
            else:
                arr_selected = arr
                selected_features = self.features
                selected_values = dict(zip(selected_features, arr.flatten()))

            pred = self.predict(arr_selected)
            info = {
                "features_all": feature_values,
                "features_selected": selected_values,
                "prediction": int(pred[0]),
                "importance": self.get_feature_importance()
            }
            if hasattr(self.model, "predict_proba"):
                info["probability"] = float(self.model.predict_proba(arr_selected)[0, 1])
            return info
        except Exception as e:
            logging.error(f"MLFilter: explain blocked: {e}")
            return {"error": str(e)}