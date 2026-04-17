"""
Page 4: Model Performance
View accuracy metrics, confusion matrices, and feature importance for all models.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_manager import load_all_models, load_training_data
from utils.ratio_engine import RATIO_NAMES, RATIO_LABELS
from utils.charts import confusion_matrix_heatmap, feature_importance_chart

st.set_page_config(page_title="Model Performance | Financial DNA", page_icon="📊", layout="wide")

st.markdown("# 📊 Model Performance")
st.markdown("Compare accuracy, confusion matrices, and feature importance across all ML models.")
st.markdown("---")

@st.cache_resource
def get_models():
    return load_all_models()

@st.cache_data
def get_training_df():
    return load_training_data()

try:
    models = get_models()
    training_df = get_training_df()
    metrics = models["metrics"]
    le = models["label_encoder"]
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# ── Model Comparison Table ──
st.markdown("### Model Accuracy Comparison")

model_names = {
    "rf": "Random Forest",
    "xgb": "XGBoost",
    "logreg": "Logistic Regression",
    "knn": "K-Nearest Neighbors",
    "svc": "SVC (rbf)",
    "ensemble": "Voting Ensemble",
}

summary_data = []
for key, name in model_names.items():
    if key in metrics:
        m = metrics[key]
        row = {
            "Model": name,
            "Test Acc.": f"{m['accuracy']:.1%}",
            "Balanced Acc.": f"{m.get('balanced_accuracy', m['accuracy']):.1%}",
            "F1 (weighted)": f"{m['f1_weighted']:.1%}",
            "Accuracy (raw)": m["accuracy"],
        }
        if "cv_mean" in m:
            row["5-fold CV"] = f"{m['cv_mean']:.1%} ± {m['cv_std']:.1%}"
        summary_data.append(row)

summary_df = pd.DataFrame(summary_data)

# Highlight best model
best_idx = summary_df["Accuracy (raw)"].idxmax()
display_cols = [c for c in
    ["Model", "Test Acc.", "Balanced Acc.", "F1 (weighted)", "5-fold CV"]
    if c in summary_df.columns]
st.dataframe(
    summary_df[display_cols],
    hide_index=True,
    use_container_width=True,
)

best_model = summary_df.loc[best_idx, "Model"]
best_acc = summary_df.loc[best_idx, "Test Acc."]
st.success(f"**Best performing model:** {best_model} with {best_acc} accuracy")

st.markdown("""
> **Note:** These models classify companies into 11 GICS sectors using only 10 financial
> ratios. A random baseline would achieve ~9% accuracy. The models significantly outperform
> chance, demonstrating that financial statement structure carries meaningful sector signal.
""")

st.markdown("---")

# ── Confusion Matrix ──
st.markdown("### Confusion Matrix")

model_select = st.selectbox(
    "Select model to view",
    options=list(model_names.values()),
    index=0,
)

# Reverse lookup key
selected_key = [k for k, v in model_names.items() if v == model_select][0]

if selected_key in metrics and "confusion_matrix" in metrics[selected_key]:
    cm = metrics[selected_key]["confusion_matrix"]
    labels = list(le.classes_)

    fig = confusion_matrix_heatmap(cm, labels, title=f"Confusion Matrix — {model_select}")
    st.plotly_chart(fig, use_container_width=True)

    # Interpretation
    st.markdown("#### Interpretation")

    # Find most confused pairs
    cm_off_diag = cm.copy().astype(float)
    np.fill_diagonal(cm_off_diag, 0)

    if cm_off_diag.sum() > 0:
        max_idx = np.unravel_index(cm_off_diag.argmax(), cm_off_diag.shape)
        true_label = labels[max_idx[0]]
        pred_label = labels[max_idx[1]]
        count = int(cm_off_diag[max_idx])
        st.info(
            f"**Most common confusion:** {true_label} companies misclassified as {pred_label} "
            f"({count} instances). This makes sense because these sectors can share similar "
            f"financial structures."
        )
    else:
        st.info("No misclassifications in the test set for this model.")

st.markdown("---")

# ── Per-Sector Performance ──
st.markdown("### Per-Sector Classification Report")

if selected_key in metrics and "classification_report" in metrics[selected_key]:
    report = metrics[selected_key]["classification_report"]

    report_data = []
    for sector in le.classes_:
        if sector in report:
            r = report[sector]
            report_data.append({
                "Sector": sector,
                "Precision": f"{r['precision']:.2f}",
                "Recall": f"{r['recall']:.2f}",
                "F1 Score": f"{r['f1-score']:.2f}",
                "Support": int(r["support"]),
            })

    report_df = pd.DataFrame(report_data)
    st.dataframe(report_df, hide_index=True, use_container_width=True)

    # Highlight best and worst sectors
    report_df["f1_raw"] = [report[s]["f1-score"] for s in le.classes_ if s in report]
    best_sector = report_df.loc[report_df["f1_raw"].idxmax(), "Sector"]
    worst_sector = report_df.loc[report_df["f1_raw"].idxmin(), "Sector"]

    col1, col2 = st.columns(2)
    with col1:
        st.success(f"**Easiest to classify:** {best_sector}")
    with col2:
        st.warning(f"**Hardest to classify:** {worst_sector}")

st.markdown("---")

# ── Feature Importance ──
st.markdown("### Feature Importance")

fi_tabs = st.tabs(["Random Forest", "XGBoost", "Logistic Regression"])

for i, (key, name) in enumerate([("rf", "Random Forest"), ("xgb", "XGBoost"), ("logreg", "Logistic Regression")]):
    with fi_tabs[i]:
        if key in metrics and "feature_importance" in metrics[key]:
            fig = feature_importance_chart(
                metrics[key]["feature_importance"],
                title=f"Feature Importance — {name}"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Top 3 features
            fi = metrics[key]["feature_importance"]
            top_indices = np.argsort(fi)[::-1][:3]
            top_features = [RATIO_LABELS[j] for j in top_indices]
            st.info(f"**Top 3 most important features:** {', '.join(top_features)}")
        else:
            st.info(f"Feature importance not available for {name}.")

st.markdown("---")

# ── Explainability Section ──
st.markdown("### Model Explainability")

st.markdown("""
#### Why Does This Work?

Different industries have fundamentally different financial structures:

- **Technology** companies tend to have high gross margins (~65%), low inventory, and minimal PP&E
  (asset-light business models).
- **Utilities** have heavy PP&E investment (~55% of assets), high debt-to-assets ratios, and
  steady but moderate margins.
- **Financials** are unique: very high debt-to-assets (leverage is the business), minimal
  inventory, and low PP&E.
- **Energy** companies carry significant PP&E (oil rigs, refineries), high COGS, and
  volatile net margins.
- **Consumer Staples** show moderate and stable margins, meaningful inventory, and
  moderate PP&E.

These patterns are consistent enough that ML models can learn the "financial DNA" of
each sector from just 10 ratios.

#### When Might It Fail?

- **Conglomerates** that span multiple sectors (e.g., Amazon: tech + retail + cloud)
- **Companies in transition** (e.g., a retailer pivoting to digital)
- **Financial companies** often confuse models due to their unusual balance sheet structure
- **Sector boundaries** aren't always clean — healthcare services may look like consumer companies
""")
