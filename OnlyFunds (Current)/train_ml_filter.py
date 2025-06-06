import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt

try:
    from xgboost import XGBClassifier
    xgb_available = True
except ImportError:
    xgb_available = False

CSV_PATH = "ml_training_data.csv"
MODEL_PATH = "ml_filter_model.pkl"
TARGET_COL = "label"

def safe_input(prompt, default="y"):
    try:
        return input(prompt)
    except EOFError:
        return default

def automatic_feature_selection(X, y):
    print("[ML_TRAINER] Selecting most predictive features using RandomForest...")
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    rf.fit(X, y)
    selector = SelectFromModel(rf, prefit=True, threshold='median')
    X_selected = selector.transform(X)
    selected_features = X.columns[selector.get_support()]
    print(f"[ML_TRAINER] Selected {len(selected_features)} features out of {X.shape[1]}.")
    return X_selected, selected_features, selector

def hyperparameter_search(model, param_grid, X, y):
    print(f"[ML_TRAINER] Running GridSearchCV for {type(model).__name__}...")
    grid = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid.fit(X, y)
    print(f"[ML_TRAINER] Best params: {grid.best_params_}")
    return grid.best_estimator_

def compare_models(X_train, X_test, y_train, y_test):
    results = {}
    # Logistic Regression
    print("\n[ML_TRAINER] Training LogisticRegression...")
    logreg_params = {"C": [0.01, 0.1, 1.0, 10.0], "solver": ["lbfgs"], "max_iter": [1000]}
    logreg = hyperparameter_search(LogisticRegression(), logreg_params, X_train, y_train)
    results["LogisticRegression"] = logreg

    # Random Forest
    print("\n[ML_TRAINER] Training RandomForestClassifier...")
    rf_params = {"n_estimators": [100, 200], "max_depth": [None, 3, 6]}
    rf = hyperparameter_search(RandomForestClassifier(), rf_params, X_train, y_train)
    results["RandomForest"] = rf

    # XGBoost
    if xgb_available:
        print("\n[ML_TRAINER] Training XGBoost...")
        xgb_params = {"n_estimators": [100, 200], "max_depth": [3, 6], "learning_rate": [0.1, 0.3]}
        xgb = hyperparameter_search(XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgb_params, X_train, y_train)
        results["XGBoost"] = xgb

    print("\n[ML_TRAINER] Model comparison:")
    for name, model in results.items():
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        try:
            auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        except Exception:
            auc = None
        print(f"{name}: Accuracy = {acc:.3f}, ROC AUC = {auc if auc is not None else 'n/a'}")
    return results

def evaluate_model(model, X_test, y_test, model_name="Model"):
    preds = model.predict(X_test)
    print(f"\n[ML_TRAINER] Evaluation for {model_name}:")
    print(classification_report(y_test, preds))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, preds))
    try:
        if hasattr(model, "predict_proba"):
            auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
            print("ROC AUC: {:.3f}".format(auc))
    except Exception:
        pass
    try:
        plt.figure(figsize=(5, 5))
        plt.title(f"{model_name} - Confusion Matrix")
        cm = confusion_matrix(y_test, preds)
        plt.imshow(cm, cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.colorbar()
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="red")
        plt.show()
    except Exception:
        pass

def main():
    print("[ML_TRAINER] Loading training data...")
    df = pd.read_csv(CSV_PATH)
    print(f"[ML_TRAINER] Loaded {len(df)} rows.")

    df = df.dropna(subset=[TARGET_COL])
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])

    # Remove non-numeric columns
    X = X.select_dtypes(include=[np.number])
    print(f"[ML_TRAINER] Using {X.shape[1]} numeric features.")

    # Feature selection
    X_selected, selected_features, selector = automatic_feature_selection(X, y)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42, stratify=y)

    # Model comparison
    results = compare_models(X_train, X_test, y_train, y_test)

    # Let user pick best model
    print("\nAvailable models:", ", ".join(results.keys()))
    best_model_name = safe_input("Enter model to use (default: LogisticRegression): ", default="LogisticRegression")
    best_model_name = best_model_name or "LogisticRegression"
    model = results[best_model_name]

    evaluate_model(model, X_test, y_test, model_name=best_model_name)
    print(f"\n[ML_TRAINER] Selected model: {best_model_name}")

    # Save selector and model together (as tuple)
    if os.path.exists(MODEL_PATH):
        confirm = safe_input(f"Model file {MODEL_PATH} exists. Overwrite? [y/N]: ", default="n").lower()
        if confirm != "y":
            print("[ML_TRAINER] Model not overwritten. Exiting.")
            return

    joblib.dump((model, selector, list(selected_features)), MODEL_PATH)
    print(f"[ML_TRAINER] Model and feature selector saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()