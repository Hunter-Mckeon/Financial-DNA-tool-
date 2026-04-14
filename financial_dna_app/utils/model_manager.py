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
    artifacts = {}
    try:
        artifacts["rf"] = load_artifact("rf_model.pkl")
        artifacts["xgb"] = load_artifact("xgb_model.pkl")
        artifacts["logreg"] = load_artifact("logreg_model.pkl")
        artifacts["knn"] = load_artifact("knn_model.pkl")
        artifacts["scaler"] = load_artifact("scaler.pkl")
        artifacts["label_encoder"] = load_artifact("label_encoder.pkl")
        artifacts["metrics"] = load_artifact("metrics.pkl")
        artifacts["pca"] = load_artifact("pca_model.pkl")
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Model file not found: {e}. Run scripts/train_models.py first."
        )
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
    """
    vec = np.array(ratios_to_vector(ratios)).reshape(1, -1)
    vec = np.nan_to_num(vec, nan=0.0)
    vec_scaled = scaler.transform(vec)

    # Scale all training data
    features = training_df[RATIO_NAMES].fillna(0).values
    features_scaled = scaler.transform(features)

    # Compute distances
    distances = np.linalg.norm(features_scaled - vec_scaled, axis=1)

    # Get top N
    top_indices = np.argsort(distances)[:n_peers]

    result = training_df.iloc[top_indices][
        ["ticker", "company_name", "sector", "industry"]
    ].copy()
    result["distance"] = distances[top_indices]
    result["rank"] = range(1, n_peers + 1)
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
