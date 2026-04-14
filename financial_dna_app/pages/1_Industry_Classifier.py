"""
Page 1: Industry Classifier
Enter a stock ticker → predict its industry sector using ML models.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_manager import (
    load_all_models, load_training_data, predict_industry, get_industry_avg_ratios
)
from utils.data_fetcher import fetch_company_ratios_live
from utils.ratio_engine import RATIO_NAMES, RATIO_LABELS
from utils.charts import radar_chart, feature_importance_chart

st.set_page_config(page_title="Industry Classifier | Financial DNA", page_icon="🏭", layout="wide")

st.markdown("# 🏭 Industry Classifier")
st.markdown("Predict a company's industry sector from its financial DNA — 10 common-size ratios.")
st.markdown("---")

# Load models
@st.cache_resource
def get_models():
    return load_all_models()

@st.cache_data
def get_training_df():
    return load_training_data()

try:
    models = get_models()
    training_df = get_training_df()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# ── User Inputs ──
col_input1, col_input2, col_input3 = st.columns([2, 1, 1])

with col_input1:
    ticker_input = st.text_input(
        "Enter a stock ticker",
        value="AAPL",
        placeholder="e.g., AAPL, MSFT, XOM",
        help="Enter any publicly traded stock ticker symbol"
    )

with col_input2:
    model_choice = st.selectbox(
        "Select ML Model",
        options=["Random Forest", "XGBoost", "Logistic Regression"],
        index=0,
    )

with col_input3:
    st.markdown("<br>", unsafe_allow_html=True)
    run_button = st.button("🔍 Analyze", type="primary", use_container_width=True)

# Map model choice to key
model_map = {
    "Random Forest": "rf",
    "XGBoost": "xgb",
    "Logistic Regression": "logreg",
}

if run_button and ticker_input:
    ticker = ticker_input.strip().upper()

    with st.spinner(f"Fetching financial data for {ticker}..."):
        # First try to find in training data
        match = training_df[training_df["ticker"] == ticker]

        if not match.empty:
            row = match.iloc[0]
            company_name = row["company_name"]
            actual_sector = row["sector"]
            actual_industry = row["industry"]
            ratios = {r: row[r] for r in RATIO_NAMES}
            data_source = "training database"
        else:
            # Try yfinance live fetch
            live_data = fetch_company_ratios_live(ticker)
            if live_data is None:
                st.error(
                    f"Could not fetch data for **{ticker}**. "
                    "Make sure it's a valid ticker and you have internet access. "
                    "You can also try a ticker from the training set."
                )
                st.markdown("**Sample tickers to try:** AAPL, MSFT, AMZN, GOOGL, JPM, XOM, JNJ, PG, NEE, CAT")
                st.stop()
            company_name = live_data["company_name"]
            actual_sector = live_data["sector"]
            actual_industry = live_data["industry"]
            ratios = {r: live_data[r] for r in RATIO_NAMES}
            data_source = "yfinance (live)"

    # Run prediction
    model_key = model_map[model_choice]
    model = models[model_key]
    scaler = models["scaler"]
    le = models["label_encoder"]

    predicted_sector, probabilities = predict_industry(ratios, model, scaler, le)
    is_correct = predicted_sector == actual_sector

    # ── Results ──
    st.markdown("---")
    st.markdown(f"## Results for {company_name} ({ticker})")
    st.caption(f"Data source: {data_source}")

    # Prediction summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Predicted Sector", predicted_sector)
    with col2:
        st.metric("Actual Sector", actual_sector)
    with col3:
        if is_correct:
            st.success("✅ Correct Prediction!", icon="✅")
        else:
            st.warning("❌ Misclassified", icon="⚠️")

    # Confidence
    if probabilities:
        top_confidence = probabilities.get(predicted_sector, 0)
        st.markdown(f"**Confidence:** {top_confidence:.1%}")

        # Show top 5 probabilities
        st.markdown("#### Prediction Probabilities")
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:5]
        prob_df = pd.DataFrame(sorted_probs, columns=["Sector", "Probability"])
        prob_df["Probability"] = prob_df["Probability"].map("{:.1%}".format)
        st.dataframe(prob_df, hide_index=True, use_container_width=True)

    st.markdown("---")

    # Radar chart: company vs industry average
    col_chart1, col_chart2 = st.columns([3, 2])

    with col_chart1:
        industry_avg = get_industry_avg_ratios(training_df, predicted_sector)
        fig = radar_chart(
            ratios, industry_avg,
            company_name=f"{ticker}",
            industry_name=f"{predicted_sector} Avg",
            title=f"{ticker} vs {predicted_sector} Average"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_chart2:
        # Ratio values table
        st.markdown("#### Financial DNA Ratios")
        ratio_df = pd.DataFrame({
            "Ratio": RATIO_LABELS,
            f"{ticker}": [ratios.get(r, 0) for r in RATIO_NAMES],
            f"{predicted_sector} Avg": [industry_avg.get(r, 0) for r in RATIO_NAMES],
        })
        ratio_df[f"{ticker}"] = ratio_df[f"{ticker}"].map("{:.3f}".format)
        ratio_df[f"{predicted_sector} Avg"] = ratio_df[f"{predicted_sector} Avg"].map("{:.3f}".format)
        st.dataframe(ratio_df, hide_index=True, use_container_width=True)

    # Feature importance
    st.markdown("---")
    st.markdown("#### What Ratios Mattered Most?")
    metrics = models["metrics"]
    if "feature_importance" in metrics.get(model_key, {}):
        fig_imp = feature_importance_chart(
            metrics[model_key]["feature_importance"],
            title=f"Feature Importance — {model_choice}"
        )
        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.info("Feature importance not available for this model.")

else:
    # Show example tickers when page first loads
    st.info("👆 Enter a ticker above and click **Analyze** to classify its industry.")
    st.markdown("**Try these tickers:** AAPL (Tech), JPM (Financials), XOM (Energy), JNJ (Health Care), NEE (Utilities)")
