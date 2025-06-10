import pandas as pd
import numpy as np
import joblib
import os
import time
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from datetime import datetime
from core.features import add_all_features

try:
    from xgboost import XGBClassifier
    xgb_available = True
except ImportError:
    xgb_available = False

CSV_PATH = "ml_training_data.csv"
PIPELINE_PATH = "ml_filter_pipeline.pkl"
MODEL_MAX_AGE_HOURS = 24
TARGET_COL = "label"

NUMERIC_FEATURES = [
    'open', 'high', 'low', 'close', 'volume', 'tr', 'atr14', 'log_return', 'realized_vol_10', 'return_3',
    'roll_close_std_5', 'roll_vol_mean_5', 'roll_vol_std_5',
    'roll_close_std_10', 'roll_vol_mean_10', 'roll_vol_std_10',
    'roll_close_std_20', 'roll_vol_mean_20', 'roll_vol_std_20',
    'entry_idx', 'exit_idx', 'pnl'
]

def is_model_fresh(path, max_age_hours):
    if not os.path.exists(path):
        return False
    age_seconds = time.time() - os.path.getmtime(path)
    return age_seconds < max_age_hours * 3600

def feature_selector(X, y):
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    rf.fit(X, y)
    selector = SelectFromModel(rf, prefit=True, threshold='median')
    print(f"[TRAIN] Selected {selector.get_support().sum()} of {X.shape[1]} features.")
    return selector

def tune_model(name, model, param_grid, X, y):
    print(f"[TRAIN] Tuning {name}...")
    grid = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid.fit(X, y)
    print(f"[TRAIN] Best {name} params: {grid.best_params_}")
    return grid.best_estimator_

def evaluate_model(model, X_test, y_test, name):
    preds = model.predict(X_test)
    print(f"\n[EVAL] {name}")
    print(classification_report(y_test, preds))
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))
    if hasattr(model, "predict_proba"):
        try:
            auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
            print("ROC AUC:", round(auc, 4))
        except:
            pass

def train_pipeline(force=False):
    if not force and is_model_fresh(PIPELINE_PATH, MODEL_MAX_AGE_HOURS):
        print("[TRAIN] Model is fresh. Skipping retraining.")
        return

    df = pd.read_csv(CSV_PATH)
    print(f"[TRAIN] Loaded {len(df)} rows.")
    df = add_all_features(df)

    used_features = [col for col in NUMERIC_FEATURES if col in df.columns]
    df = df.dropna(subset=[TARGET_COL])
    y = df[TARGET_COL]
    X = df[used_features].select_dtypes(include=[np.number])

    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.dropna(inplace=True)
    y = y.loc[X.index]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    selector = feature_selector(X, y)
    X_sel = selector.transform(X_scaled)

    X_train, X_test, y_train, y_test = train_test_split(
        X_sel, y, test_size=0.2, stratify=y, random_state=42
    )

    models = {
        "LogisticRegression": tune_model(
            "LogisticRegression", LogisticRegression(max_iter=2000),
            {"C": [0.1, 1.0, 10.0]}, X_train, y_train
        ),
        "RandomForest": tune_model(
            "RandomForest", RandomForestClassifier(),
            {"n_estimators": [100], "max_depth": [None, 3, 6]}, X_train, y_train
        )
    }

    if xgb_available:
        models["XGBoost"] = tune_model(
            "XGBoost", XGBClassifier(eval_metric="logloss", use_label_encoder=False),
            {"n_estimators": [100], "max_depth": [3], "learning_rate": [0.1]}, X_train, y_train
        )

    for name, model in models.items():
        evaluate_model(model, X_test, y_test, name)

    # Automatically select model with highest ROC AUC
    best_model = None
    best_auc = -1
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            try:
                auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
                print(f"[SELECT] {name} AUC: {auc:.4f}")
                if auc > best_auc:
                    best_model = model
                    best_auc = auc
            except:
                continue

    best_name = [k for k, v in models.items() if v == best_model][0]
    print(f"[SELECT] Auto-selected best model: {best_name} (AUC: {best_auc:.4f})")

    pipeline = Pipeline([
        ("scaler", scaler),
        ("selector", selector),
        ("classifier", best_model)
    ])

    joblib.dump(pipeline, PIPELINE_PATH)
    print(f"[TRAIN] ✅ Saved pipeline to {PIPELINE_PATH}")

if __name__ == "__main__":
    train_pipeline()
