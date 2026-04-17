"""
Page 3: Company Comparison
Compare two companies' financial DNA side by side.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_manager import (
    load_all_models, load_training_data, predict_industry
)
from utils.data_fetcher import fetch_company_ratios_live
from utils.ratio_engine import RATIO_NAMES, RATIO_LABELS
from utils.charts import dual_radar_chart

st.set_page_config(page_title="Company Comparison | Financial DNA", page_icon="⚖️", layout="wide")

st.markdown("# ⚖️ Company Comparison")
st.markdown("Compare the financial DNA of any two companies head-to-head.")
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
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()


def get_company_data(ticker):
    """Fetch company data from training set or yfinance."""
    match = training_df[training_df["ticker"] == ticker]
    if not match.empty:
        row = match.iloc[0]
        return {
            "ticker": ticker,
            "company_name": row["company_name"],
            "sector": row["sector"],
            "industry": row["industry"],
            "ratios": {r: row[r] for r in RATIO_NAMES},
            "source": "database",
        }
    else:
        live = fetch_company_ratios_live(ticker)
        if live is None:
            return None
        return {
            "ticker": ticker,
            "company_name": live["company_name"],
            "sector": live["sector"],
            "industry": live["industry"],
            "ratios": {r: live[r] for r in RATIO_NAMES},
            "source": "yfinance",
        }


# ── User Inputs ──
col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    ticker_a = st.text_input("Company A", value="AAPL", placeholder="e.g., AAPL")

with col2:
    ticker_b = st.text_input("Company B", value="MSFT", placeholder="e.g., MSFT")

with col3:
    st.markdown("<br>", unsafe_allow_html=True)
    run_button = st.button("⚖️ Compare", type="primary", use_container_width=True)

if run_button and ticker_a and ticker_b:
    ta = ticker_a.strip().upper()
    tb = ticker_b.strip().upper()

    with st.spinner(f"Comparing {ta} vs {tb}..."):
        data_a = get_company_data(ta)
        data_b = get_company_data(tb)

    if data_a is None:
        st.error(f"Could not fetch data for **{ta}**.")
        st.stop()
    if data_b is None:
        st.error(f"Could not fetch data for **{tb}**.")
        st.stop()

    # ── Company Info ──
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"### {data_a['company_name']} ({ta})")
        st.markdown(f"**Sector:** {data_a['sector']}")
        st.markdown(f"**Industry:** {data_a['industry']}")

        # Prediction
        pred_a, _ = predict_industry(
            data_a["ratios"], models["rf"], models["scaler"], models["label_encoder"]
        )
        is_correct_a = pred_a == data_a["sector"]
        st.markdown(f"**ML Predicted Sector:** {pred_a} {'✅' if is_correct_a else '❌'}")

    with col2:
        st.markdown(f"### {data_b['company_name']} ({tb})")
        st.markdown(f"**Sector:** {data_b['sector']}")
        st.markdown(f"**Industry:** {data_b['industry']}")

        pred_b, _ = predict_industry(
            data_b["ratios"], models["rf"], models["scaler"], models["label_encoder"]
        )
        is_correct_b = pred_b == data_b["sector"]
        st.markdown(f"**ML Predicted Sector:** {pred_b} {'✅' if is_correct_b else '❌'}")

    # ── Distance Score ──
    scaler = models["scaler"]
    vec_a = np.array([data_a["ratios"].get(r, 0) for r in RATIO_NAMES]).reshape(1, -1)
    vec_b = np.array([data_b["ratios"].get(r, 0) for r in RATIO_NAMES]).reshape(1, -1)
    vec_a = np.nan_to_num(vec_a, nan=0.0)
    vec_b = np.nan_to_num(vec_b, nan=0.0)
    vec_a_scaled = scaler.transform(vec_a)
    vec_b_scaled = scaler.transform(vec_b)
    distance = np.linalg.norm(vec_a_scaled - vec_b_scaled)

    st.markdown("---")

    # Similarity gauge
    max_distance = 10.0  # rough max
    similarity = max(0, 1 - distance / max_distance) * 100

    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric("Euclidean Distance", f"{distance:.3f}")
    with col_m2:
        st.metric("Similarity Score", f"{similarity:.0f}%")
    with col_m3:
        same_sector = data_a["sector"] == data_b["sector"]
        st.metric("Same Sector?", "Yes ✅" if same_sector else "No ❌")

    st.markdown("---")

    # ── Dual Radar Chart ──
    fig = dual_radar_chart(
        data_a["ratios"], data_b["ratios"],
        name_a=f"{ta} ({data_a['sector']})",
        name_b=f"{tb} ({data_b['sector']})",
        title=f"Financial DNA: {ta} vs {tb}",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Detailed Comparison Table ──
    st.markdown("### Ratio-by-Ratio Comparison")

    comparison_data = []
    for i, r in enumerate(RATIO_NAMES):
        val_a = data_a["ratios"].get(r, 0)
        val_b = data_b["ratios"].get(r, 0)
        delta = val_a - val_b
        comparison_data.append({
            "Ratio": RATIO_LABELS[i],
            ta: f"{val_a:.4f}",
            tb: f"{val_b:.4f}",
            "Delta": f"{delta:+.4f}",
            "Abs Delta": abs(delta),
        })

    comp_df = pd.DataFrame(comparison_data)

    # Highlight biggest differences
    st.dataframe(
        comp_df[["Ratio", ta, tb, "Delta"]],
        hide_index=True,
        use_container_width=True,
    )

    # Biggest differences callout
    comp_df_sorted = comp_df.sort_values("Abs Delta", ascending=False)
    top_diff = comp_df_sorted.iloc[0]
    st.info(
        f"**Biggest difference:** {top_diff['Ratio']} "
        f"({ta}: {top_diff[ta]}, {tb}: {top_diff[tb]}, Δ: {top_diff['Delta']})"
    )

else:
    st.info("👆 Enter two tickers and click **Compare** to see their financial DNA side by side.")
    st.markdown("**Try:** AAPL vs MSFT, XOM vs NEE, JPM vs COST, AMZN vs WMT")
