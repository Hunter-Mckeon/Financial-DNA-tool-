"""
train_models.py — Train Random Forest, XGBoost, Logistic Regression, and KNN
on the company ratios dataset, then save all model artifacts.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score
)
from sklearn.decomposition import PCA
from xgboost import XGBClassifier

from utils.ratio_engine import RATIO_NAMES

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def save_artifact(obj, filename):
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, filename)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"  Saved: {filename}")


def train_models():
    print("=" * 60)
    print("Financial DNA — Training ML Models")
    print("=" * 60)

    # Load data
    data_path = os.path.join(DATA_DIR, "company_ratios.csv")
    df = pd.read_csv(data_path)
    print(f"\nLoaded {len(df)} companies from {data_path}")

    # Prepare features and labels
    X = df[RATIO_NAMES].fillna(0.0).values
    y_sector = df["sector"].values

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y_sector)
    n_classes = len(le.classes_)
    print(f"Target classes (sectors): {n_classes}")
    print(f"Classes: {list(le.classes_)}")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

    # PCA for visualization (fit on all data)
    pca = PCA(n_components=2, random_state=42)
    pca.fit(X_scaled)
    print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")

    metrics = {}

    # --- 1. Random Forest ---
    print("\n" + "-" * 40)
    print("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_f1 = f1_score(y_test, rf_pred, average="weighted")
    print(f"  Accuracy: {rf_acc:.4f}")
    print(f"  F1 (weighted): {rf_f1:.4f}")
    metrics["rf"] = {
        "accuracy": rf_acc,
        "f1_weighted": rf_f1,
        "confusion_matrix": confusion_matrix(y_test, rf_pred),
        "classification_report": classification_report(
            y_test, rf_pred, target_names=le.classes_, output_dict=True
        ),
        "feature_importance": rf.feature_importances_.tolist(),
    }

    # --- 2. XGBoost ---
    print("\n" + "-" * 40)
    print("Training XGBoost...")
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="mlogloss",
        use_label_encoder=False,
    )
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    xgb_acc = accuracy_score(y_test, xgb_pred)
    xgb_f1 = f1_score(y_test, xgb_pred, average="weighted")
    print(f"  Accuracy: {xgb_acc:.4f}")
    print(f"  F1 (weighted): {xgb_f1:.4f}")
    metrics["xgb"] = {
        "accuracy": xgb_acc,
        "f1_weighted": xgb_f1,
        "confusion_matrix": confusion_matrix(y_test, xgb_pred),
        "classification_report": classification_report(
            y_test, xgb_pred, target_names=le.classes_, output_dict=True
        ),
        "feature_importance": xgb.feature_importances_.tolist(),
    }

    # --- 3. Logistic Regression ---
    print("\n" + "-" * 40)
    print("Training Logistic Regression (baseline)...")
    logreg = LogisticRegression(
        max_iter=2000,
        multi_class="multinomial",
        solver="lbfgs",
        class_weight="balanced",
        random_state=42,
    )
    logreg.fit(X_train, y_train)
    lr_pred = logreg.predict(X_test)
    lr_acc = accuracy_score(y_test, lr_pred)
    lr_f1 = f1_score(y_test, lr_pred, average="weighted")
    print(f"  Accuracy: {lr_acc:.4f}")
    print(f"  F1 (weighted): {lr_f1:.4f}")
    metrics["logreg"] = {
        "accuracy": lr_acc,
        "f1_weighted": lr_f1,
        "confusion_matrix": confusion_matrix(y_test, lr_pred),
        "classification_report": classification_report(
            y_test, lr_pred, target_names=le.classes_, output_dict=True
        ),
        "feature_importance": np.abs(logreg.coef_).mean(axis=0).tolist(),
    }

    # --- 4. KNN (for peer finding) ---
    print("\n" + "-" * 40)
    print("Training KNN...")
    knn = KNeighborsClassifier(
        n_neighbors=10,
        weights="distance",
        metric="euclidean",
        n_jobs=-1,
    )
    knn.fit(X_train, y_train)
    knn_pred = knn.predict(X_test)
    knn_acc = accuracy_score(y_test, knn_pred)
    knn_f1 = f1_score(y_test, knn_pred, average="weighted")
    print(f"  Accuracy: {knn_acc:.4f}")
    print(f"  F1 (weighted): {knn_f1:.4f}")
    metrics["knn"] = {
        "accuracy": knn_acc,
        "f1_weighted": knn_f1,
        "confusion_matrix": confusion_matrix(y_test, knn_pred),
        "classification_report": classification_report(
            y_test, knn_pred, target_names=le.classes_, output_dict=True
        ),
    }

    # --- Save all artifacts ---
    print("\n" + "-" * 40)
    print("Saving model artifacts...")
    save_artifact(rf, "rf_model.pkl")
    save_artifact(xgb, "xgb_model.pkl")
    save_artifact(logreg, "logreg_model.pkl")
    save_artifact(knn, "knn_model.pkl")
    save_artifact(scaler, "scaler.pkl")
    save_artifact(le, "label_encoder.pkl")
    save_artifact(metrics, "metrics.pkl")
    save_artifact(pca, "pca_model.pkl")

    # Print summary
    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"{'Model':<25} {'Accuracy':>10} {'F1 (weighted)':>15}")
    print("-" * 50)
    print(f"{'Random Forest':<25} {rf_acc:>10.4f} {rf_f1:>15.4f}")
    print(f"{'XGBoost':<25} {xgb_acc:>10.4f} {xgb_f1:>15.4f}")
    print(f"{'Logistic Regression':<25} {lr_acc:>10.4f} {lr_f1:>15.4f}")
    print(f"{'KNN (k=10)':<25} {knn_acc:>10.4f} {knn_f1:>15.4f}")
    print("=" * 60)
    print("Training complete!")


if __name__ == "__main__":
    train_models()
