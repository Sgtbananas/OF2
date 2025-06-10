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
    - Bulletproof feature/selector alignment: no more axis errors.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model: Optional[Any] = None
        self.selector: Optional[Any] = None
        self.features: Optional[List[str]] = None
        self.model_path = model_path

        if model_path and os.path.exists(model_path):
            loaded = joblib.load(model_path)
            # PATCH: Handle both dict and tuple formats for robustness
            if isinstance(loaded, tuple):
                self.model = loaded[0]
                if len(loaded) > 1:
                    self.selector = loaded[1]
                if len(loaded) > 2:
                    self.features = loaded[2]
                logging.info(f"MLFilter: Loaded model, selector, features from {model_path}")
            elif isinstance(loaded, dict):
                self.model = loaded.get("model", None)
                self.selector = loaded.get("selector", loaded.get("feature_selector", None)) or loaded.get("model", None)
                # THIS IS THE CRUCIAL LINE: use the full feature list from training:
                self.features = loaded.get("full_feature_list", loaded.get("features", None))
                if self.features is None:
                    self.features = loaded.get("selected_features", None)
                logging.info(f"MLFilter: Loaded model dict from {model_path}")
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
        BULLETPROOF: Ensures DataFrame passed to selector has the exact columns, order, and length as at training time.
        """
        if self.features is None:
            raise RuntimeError("MLFilter: Model loaded without feature names. Cannot extract features.")
        # Fill missing features with 0.0, drop extras, and order
        for col in self.features:
            if col not in df.columns:
                df[col] = 0.0
        df = df[self.features]
        row = df.iloc[[idx]].values  # 2D for selector
        if self.selector is not None:
            try:
                support_mask = self.selector.get_support()
                if row.shape[1] != len(support_mask):
                    logging.error(
                        f"MLFilter: Feature count mismatch for selector! "
                        f"Row has {row.shape[1]} columns, selector expects {len(support_mask)}"
                    )
                    logging.error(f"Selector expects features: {self.features}")
                    logging.error(f"DF columns: {df.columns.tolist()}")
                    logging.error(f"Selector support mask length: {len(support_mask)}")
                    logging.error(f"DF shape: {df.shape}")
                    raise RuntimeError(
                        f"MLFilter: Feature selector transform failed: boolean index did not match indexed array along axis 1; "
                        f"size of axis is {row.shape[1]} but size of corresponding boolean axis is {len(support_mask)}"
                    )
                row = self.selector.transform(row)
                logging.debug(f"MLFilter: Selector reduced features to shape {row.shape}")
            except Exception as e:
                logging.error(f"MLFilter: Feature selector transform failed: {e}")
                raise
        return row

    def predict(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.model, "predict"):
            return self.model.predict(X)
        raise RuntimeError("MLFilter: Model lacks predict(). Refusing to trade.")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        raise RuntimeError("MLFilter: Model lacks predict_proba().")

    def should_enter(self, df: pd.DataFrame, idx: int, signal: Any, threshold: float = 0.5) -> bool:
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
        try:
            for col in self.features:
                if col not in df.columns:
                    df[col] = 0.0
            df = df[self.features]
            feats = df.iloc[[idx]].values

            if self.selector is not None:
                support_mask = self.selector.get_support()
                if feats.shape[1] != len(support_mask):
                    logging.error(
                        f"MLFilter: Feature count mismatch for selector in explain! "
                        f"Row has {feats.shape[1]} columns, selector expects {len(support_mask)}"
                    )
                    raise RuntimeError("MLFilter: Feature selector shape mismatch in explain.")
                feats_selected = self.selector.transform(feats)
                selected_features = [f for f, sel in zip(self.features, support_mask) if sel]
                selected_values = dict(zip(selected_features, feats_selected.flatten()))
            else:
                feats_selected = feats
                selected_features = self.features
                selected_values = dict(zip(selected_features, feats.flatten()))

            pred = self.predict(feats_selected)
            info = {
                "features_all": dict(zip(self.features, feats.flatten())),
                "features_selected": selected_values,
                "prediction": int(pred[0]),
                "importance": self.get_feature_importance()
            }
            if hasattr(self.model, "predict_proba"):
                info["probability"] = float(self.model.predict_proba(feats_selected)[0, 1])
            return info
        except Exception as e:
            logging.error(f"MLFilter: explain blocked: {e}")
            return {"error": str(e)}