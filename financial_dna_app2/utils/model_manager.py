"""
model_manager.py — Load trained models and make predictions.
"""

import pickle
import os
import numpy as np
import pandas as pd
from utils.ratio_engine import RATIO_NAMES, ratios_to_vector

# Path to saved models
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def load_artifact(filename):
    """Load a pickled artifact from the models directory."""
    path = os.path.join(MODEL_DIR, filename)
    with open(path, "rb") as f:
        return pickle.load(f)


def load_all_models():
    """Load all trained models and preprocessing artifacts."""
    required = {
        "rf": "rf_model.pkl",
        "xgb": "xgb_model.pkl",
        "logreg": "logreg_model.pkl",
        "knn": "knn_model.pkl",
        "scaler": "scaler.pkl",
        "label_encoder": "label_encoder.pkl",
        "metrics": "metrics.pkl",
        "pca": "pca_model.pkl",
    }
    # Newer models trained by the updated pipeline; optional so the app keeps
    # working if someone reloads an older model directory.
    optional = {
        "svc": "svc_model.pkl",
        "ensemble": "ensemble_model.pkl",
    }

    artifacts = {}
    try:
        for k, fname in required.items():
            artifacts[k] = load_artifact(fname)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Model file not found: {e}. Run scripts/train_models.py first."
        )
    for k, fname in optional.items():
        try:
            artifacts[k] = load_artifact(fname)
        except FileNotFoundError:
            pass
    return artifacts


def load_training_data():
    """Load the training dataset."""
    path = os.path.join(DATA_DIR, "company_ratios.csv")
    return pd.read_csv(path)


def predict_industry(ratios: dict, model, scaler, label_encoder):
    """
    Predict industry from a ratio dict.
    Returns (predicted_label, probabilities_dict).
    """
    vec = np.array(ratios_to_vector(ratios)).reshape(1, -1)

    # Impute NaN with 0 for prediction
    vec = np.nan_to_num(vec, nan=0.0)

    vec_scaled = scaler.transform(vec)

    pred_idx = model.predict(vec_scaled)[0]
    pred_label = label_encoder.inverse_transform([pred_idx])[0]

    # Get probabilities if model supports it
    proba = {}
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(vec_scaled)[0]
        for idx, p in enumerate(probs):
            label = label_encoder.inverse_transform([idx])[0]
            proba[label] = float(p)

    return pred_label, proba


def find_peers(ratios: dict, scaler, training_df, n_peers=10):
    """
    Find the N nearest peer companies by Euclidean distance in scaled ratio space.
    Returns a DataFrame with: ticker, company_name, sector, industry, distance.

    The training data may contain multiple rows per ticker (one per fiscal year).
    We dedupe so each company appears at most once — keeping the closest year.
    """
    vec = np.array(ratios_to_vector(ratios)).reshape(1, -1)
    vec = np.nan_to_num(vec, nan=0.0)
    vec_scaled = scaler.transform(vec)

    # Scale all training data
    features = training_df[RATIO_NAMES].fillna(0).values
    features_scaled = scaler.transform(features)

    # Compute distances for every row
    distances = np.linalg.norm(features_scaled - vec_scaled, axis=1)

    # Sort ALL rows by distance, then keep the first occurrence of each ticker
    # (= the closest fiscal year for that company).
    sorted_idx = np.argsort(distances)
    seen = set()
    keep = []
    for idx in sorted_idx:
        tkr = training_df.iloc[idx]["ticker"]
        if tkr in seen:
            continue
        seen.add(tkr)
        keep.append(idx)
        if len(keep) >= n_peers:
            break

    result = training_df.iloc[keep][
        ["ticker", "company_name", "sector", "industry"]
    ].copy()
    result["distance"] = distances[keep]
    result["rank"] = range(1, len(keep) + 1)
    result = result.reset_index(drop=True)

    return result


def get_industry_avg_ratios(training_df, sector):
    """
    Compute the average ratios for a given sector from the training data.
    """
    sector_df = training_df[training_df["sector"] == sector]
    if sector_df.empty:
        return {r: 0.0 for r in RATIO_NAMES}
    return sector_df[RATIO_NAMES].mean().to_dict()
