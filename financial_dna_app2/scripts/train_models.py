"""
train_models.py — Train Random Forest, XGBoost, Logistic Regression, KNN,
and an SVC baseline on the company ratios dataset, then save all artifacts.

Key design choices (fixed vs the previous version):
  1. Scaler is fit on the TRAIN split only (no test-set leakage).
  2. PCA is also fit on TRAIN only.
  3. Hyperparameters are tuned via stratified 5-fold CV on the TRAIN split,
     then the tuned model is evaluated once on the held-out TEST split so
     reported numbers reflect true generalization.
  4. Deprecated scikit-learn / XGBoost params removed.
  5. A soft-voting ensemble is computed to see if it beats the best single model.
  6. Metrics stored include both holdout accuracy AND CV mean ± std so the
     Model Performance page can show CV error bars.
"""

import sys
import os
import warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV, cross_val_score,
    GroupShuffleSplit, StratifiedGroupKFold,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, balanced_accuracy_score,
)
from xgboost import XGBClassifier

from utils.ratio_engine import RATIO_NAMES

warnings.filterwarnings("ignore", category=UserWarning)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

RANDOM_STATE = 42
CV_SPLITS = 5


def save_artifact(obj, filename):
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, filename)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"  Saved: {filename}")


def _record_metrics(name, model, X_test, y_test, target_names, cv_scores=None, fi=None):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    f1w = f1_score(y_test, y_pred, average="weighted")
    print(f"  [{name}] holdout acc={acc:.4f}  balanced_acc={bal_acc:.4f}  f1={f1w:.4f}"
          + (f"  cv={cv_scores.mean():.4f}±{cv_scores.std():.4f}" if cv_scores is not None else ""))
    m = {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "f1_weighted": float(f1w),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(
            y_test, y_pred, target_names=target_names, output_dict=True
        ),
    }
    if cv_scores is not None:
        m["cv_mean"] = float(cv_scores.mean())
        m["cv_std"] = float(cv_scores.std())
    if fi is not None:
        m["feature_importance"] = list(map(float, fi))
    return m


def train_models():
    print("=" * 60)
    print("Financial DNA — Training ML Models")
    print("=" * 60)

    data_path = os.path.join(DATA_DIR, "company_ratios.csv")
    df = pd.read_csv(data_path)
    print(f"\nLoaded {len(df)} rows from {data_path}")

    X = df[RATIO_NAMES].fillna(0.0).values
    y_sector = df["sector"].values

    le = LabelEncoder()
    y = le.fit_transform(y_sector)
    target_names = list(le.classes_)
    print(f"Classes ({len(target_names)}): {target_names}")

    # ---- Grouped split: every fiscal year of a given ticker must go to the
    # same side of the split.  Without this, AAPL-2023 in train + AAPL-2022 in
    # test lets the model "memorize" tickers instead of learning sector
    # patterns — that's how we got a fake 97% accuracy. ----
    if "ticker" in df.columns and df["ticker"].nunique() < len(df):
        groups = df["ticker"].values
        print(f"  Using grouped split: {len(np.unique(groups))} unique companies "
              f"across {len(df)} rows")
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
        train_idx, test_idx = next(gss.split(X, y, groups=groups))
        X_train_raw, X_test_raw = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train = groups[train_idx]
        cv = StratifiedGroupKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        # cross_val_score needs groups; wrap it with a helper that passes them
        cv_kwargs = {"groups": groups_train}
    else:
        # Fallback for single-year-per-company datasets
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )
        cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        cv_kwargs = {}
    print(f"Train: {len(X_train_raw)}  Test: {len(X_test_raw)}")

    # ---- Fit scaler on TRAIN only (no test leakage) ----
    scaler = StandardScaler().fit(X_train_raw)
    X_train = scaler.transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # PCA also fit on train only (used only for visualization)
    pca = PCA(n_components=2, random_state=RANDOM_STATE).fit(X_train)
    print(f"PCA(2) explained variance (train): {pca.explained_variance_ratio_.sum():.2%}")

    metrics = {}

    # ---- 1. Random Forest (tuned) ----
    print("\n" + "-" * 40)
    print("Tuning Random Forest...")
    rf_grid = GridSearchCV(
        RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
        param_grid={
            "n_estimators": [300, 500],
            "max_depth": [None, 10],
            "min_samples_leaf": [1, 2],
        },
        cv=cv, scoring="accuracy", n_jobs=-1,
    )
    rf_grid.fit(X_train, y_train, **cv_kwargs)
    rf = rf_grid.best_estimator_
    print(f"  Best RF params: {rf_grid.best_params_}  CV acc: {rf_grid.best_score_:.4f}")
    rf_cv = cross_val_score(rf, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1, **cv_kwargs)
    metrics["rf"] = _record_metrics("RF", rf, X_test, y_test, target_names,
                                    cv_scores=rf_cv, fi=rf.feature_importances_)

    # ---- 2. XGBoost (tuned) ----
    print("\n" + "-" * 40)
    print("Tuning XGBoost...")
    xgb_grid = GridSearchCV(
        XGBClassifier(random_state=RANDOM_STATE, eval_metric="mlogloss", n_jobs=-1),
        param_grid={
            "n_estimators": [200, 400],
            "max_depth": [4, 8],
            "learning_rate": [0.05, 0.1],
        },
        cv=cv, scoring="accuracy", n_jobs=-1,
    )
    xgb_grid.fit(X_train, y_train, **cv_kwargs)
    xgb = xgb_grid.best_estimator_
    print(f"  Best XGB params: {xgb_grid.best_params_}  CV acc: {xgb_grid.best_score_:.4f}")
    xgb_cv = cross_val_score(xgb, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1, **cv_kwargs)
    metrics["xgb"] = _record_metrics("XGB", xgb, X_test, y_test, target_names,
                                     cv_scores=xgb_cv, fi=xgb.feature_importances_)

    # ---- 3. Logistic Regression (tuned) ----
    print("\n" + "-" * 40)
    print("Tuning Logistic Regression...")
    lr_grid = GridSearchCV(
        LogisticRegression(max_iter=3000, class_weight="balanced",
                           solver="lbfgs", random_state=RANDOM_STATE),
        param_grid={"C": [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]},
        cv=cv, scoring="accuracy", n_jobs=-1,
    )
    lr_grid.fit(X_train, y_train, **cv_kwargs)
    logreg = lr_grid.best_estimator_
    print(f"  Best LR params: {lr_grid.best_params_}  CV acc: {lr_grid.best_score_:.4f}")
    lr_cv = cross_val_score(logreg, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1, **cv_kwargs)
    # For multi-class LR, use mean |coef| as the importance summary
    lr_fi = np.abs(logreg.coef_).mean(axis=0)
    metrics["logreg"] = _record_metrics("LogReg", logreg, X_test, y_test, target_names,
                                        cv_scores=lr_cv, fi=lr_fi)

    # ---- 4. KNN (for peer finding + classification) ----
    print("\n" + "-" * 40)
    print("Tuning KNN...")
    knn_grid = GridSearchCV(
        KNeighborsClassifier(weights="distance", n_jobs=-1),
        param_grid={"n_neighbors": [5, 7, 10, 15], "metric": ["euclidean", "manhattan"]},
        cv=cv, scoring="accuracy", n_jobs=-1,
    )
    knn_grid.fit(X_train, y_train, **cv_kwargs)
    knn = knn_grid.best_estimator_
    print(f"  Best KNN params: {knn_grid.best_params_}  CV acc: {knn_grid.best_score_:.4f}")
    knn_cv = cross_val_score(knn, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1, **cv_kwargs)
    metrics["knn"] = _record_metrics("KNN", knn, X_test, y_test, target_names,
                                     cv_scores=knn_cv)

    # ---- 5. SVC (rbf) — new addition, often strong on ratio data ----
    print("\n" + "-" * 40)
    print("Tuning SVC (rbf)...")
    svc_grid = GridSearchCV(
        SVC(kernel="rbf", probability=True, class_weight="balanced",
            random_state=RANDOM_STATE),
        param_grid={"C": [0.5, 1.0, 2.0, 5.0], "gamma": ["scale", 0.1, 0.5]},
        cv=cv, scoring="accuracy", n_jobs=-1,
    )
    svc_grid.fit(X_train, y_train, **cv_kwargs)
    svc = svc_grid.best_estimator_
    print(f"  Best SVC params: {svc_grid.best_params_}  CV acc: {svc_grid.best_score_:.4f}")
    svc_cv = cross_val_score(svc, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1, **cv_kwargs)
    metrics["svc"] = _record_metrics("SVC", svc, X_test, y_test, target_names,
                                     cv_scores=svc_cv)

    # ---- 6. Soft-voting ensemble of all 5 models ----
    # 5 voters means plurality ties are impossible; soft voting also averages
    # the full probability distributions so KNN's coarse probabilities get
    # smoothed by the other four.
    print("\n" + "-" * 40)
    print("Training soft-voting ensemble...")
    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("xgb", xgb), ("lr", logreg),
                    ("knn", knn), ("svc", svc)],
        voting="soft", n_jobs=-1,
    )
    ensemble.fit(X_train, y_train)
    ens_cv = cross_val_score(ensemble, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1, **cv_kwargs)
    metrics["ensemble"] = _record_metrics("Ensemble", ensemble, X_test, y_test, target_names,
                                          cv_scores=ens_cv)

    # ---- Save everything ----
    print("\n" + "-" * 40)
    print("Saving model artifacts...")
    save_artifact(rf, "rf_model.pkl")
    save_artifact(xgb, "xgb_model.pkl")
    save_artifact(logreg, "logreg_model.pkl")
    save_artifact(knn, "knn_model.pkl")
    save_artifact(svc, "svc_model.pkl")
    save_artifact(ensemble, "ensemble_model.pkl")
    save_artifact(scaler, "scaler.pkl")
    save_artifact(le, "label_encoder.pkl")
    save_artifact(metrics, "metrics.pkl")
    save_artifact(pca, "pca_model.pkl")

    # ---- Final summary ----
    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE SUMMARY  (holdout 20% test set)")
    print("=" * 60)
    print(f"{'Model':<22} {'Acc':>8} {'BalAcc':>8} {'F1(w)':>8} {'CV mean±std':>16}")
    print("-" * 66)
    for key, label in [("rf", "Random Forest"),
                       ("xgb", "XGBoost"),
                       ("logreg", "Logistic Regression"),
                       ("knn", "KNN"),
                       ("svc", "SVC (rbf)"),
                       ("ensemble", "Voting Ensemble")]:
        m = metrics[key]
        cv_str = f"{m.get('cv_mean', 0):.3f}±{m.get('cv_std', 0):.3f}"
        print(f"{label:<22} {m['accuracy']:>8.4f} {m['balanced_accuracy']:>8.4f} "
              f"{m['f1_weighted']:>8.4f} {cv_str:>16}")
    print("=" * 66)
    print("Training complete!")


if __name__ == "__main__":
    train_models()
