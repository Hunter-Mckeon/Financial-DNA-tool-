"""
Page 2: Peer Finder
Enter a stock ticker → find the most similar companies by financial profile.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_manager import load_all_models, load_training_data, find_peers
from utils.data_fetcher import fetch_company_ratios_live
from utils.ratio_engine import RATIO_NAMES, RATIO_LABELS
from utils.charts import radar_chart, peer_scatter_pca

st.set_page_config(page_title="Peer Finder | Financial DNA", page_icon="🔗", layout="wide")

st.markdown("# 🔗 Peer Finder")
st.markdown("Find the companies with the most similar financial DNA to any target company.")
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

# ── User Inputs ──
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    ticker_input = st.text_input(
        "Enter a stock ticker",
        value="AAPL",
        placeholder="e.g., AAPL, TSLA, JPM",
    )

with col2:
    n_peers = st.slider("Number of peers", min_value=5, max_value=20, value=10)

with col3:
    st.markdown("<br>", unsafe_allow_html=True)
    run_button = st.button("🔍 Find Peers", type="primary", use_container_width=True)

if run_button and ticker_input:
    ticker = ticker_input.strip().upper()

    with st.spinner(f"Analyzing {ticker}..."):
        # Get company ratios
        match = training_df[training_df["ticker"] == ticker]

        if not match.empty:
            row = match.iloc[0]
            company_name = row["company_name"]
            sector = row["sector"]
            ratios = {r: row[r] for r in RATIO_NAMES}
        else:
            live_data = fetch_company_ratios_live(ticker)
            if live_data is None:
                st.error(f"Could not fetch data for **{ticker}**. Try a ticker from the training set.")
                st.markdown("**Sample tickers:** AAPL, MSFT, AMZN, GOOGL, JPM, XOM, JNJ, PG")
                st.stop()
            company_name = live_data["company_name"]
            sector = live_data["sector"]
            ratios = {r: live_data[r] for r in RATIO_NAMES}

        # Find peers
        peers_df = find_peers(ratios, models["scaler"], training_df, n_peers=n_peers)

        # Filter out the company itself if it appears
        peers_df = peers_df[peers_df["ticker"] != ticker].head(n_peers)

    # ── Results ──
    st.markdown("---")
    st.markdown(f"## Peers for {company_name} ({ticker})")
    st.markdown(f"**Sector:** {sector}")

    # Peer table
    st.markdown("### Ranked Peer Table")
    display_df = peers_df[["rank", "ticker", "company_name", "sector", "industry", "distance"]].copy()
    display_df.columns = ["Rank", "Ticker", "Company", "Sector", "Industry", "Distance Score"]
    display_df["Distance Score"] = display_df["Distance Score"].map("{:.4f}".format)

    # Color-code: same sector = green
    st.dataframe(
        display_df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Rank": st.column_config.NumberColumn("Rank", width="small"),
            "Distance Score": st.column_config.TextColumn("Distance Score", width="medium"),
        }
    )

    # Stats
    same_sector_count = (peers_df["sector"] == sector).sum()
    st.info(
        f"**{same_sector_count} of {len(peers_df)}** peers are in the same sector ({sector}). "
        f"{'Great cluster coherence!' if same_sector_count >= n_peers * 0.5 else 'Diverse peer set — this company may have a unique financial profile.'}"
    )

    st.markdown("---")

    # PCA scatter plot
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("### Financial DNA Map (PCA)")
        # Compute PCA coordinates for all companies
        pca = models["pca"]
        scaler = models["scaler"]

        all_features = training_df[RATIO_NAMES].fillna(0).values
        all_scaled = scaler.transform(all_features)
        all_pca = pca.transform(all_scaled)

        pca_df = training_df[["ticker", "sector"]].copy()
        pca_df["PC1"] = all_pca[:, 0]
        pca_df["PC2"] = all_pca[:, 1]

        fig = peer_scatter_pca(
            pca_df,
            target_ticker=ticker,
            peer_tickers=peers_df["ticker"].tolist(),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("### Top 3 Peers — Radar Comparison")
        # Mini radar charts for top 3 peers
        for i, (_, peer_row) in enumerate(peers_df.head(3).iterrows()):
            peer_ticker = peer_row["ticker"]
            peer_ratios = {r: training_df[training_df["ticker"] == peer_ticker].iloc[0][r] for r in RATIO_NAMES}

            fig = radar_chart(
                ratios, peer_ratios,
                company_name=ticker,
                industry_name=peer_ticker,
                title=f"#{i+1}: {ticker} vs {peer_ticker}"
            )
            fig.update_layout(height=350, margin=dict(t=50, b=20, l=40, r=40))
            st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👆 Enter a ticker and click **Find Peers** to discover the most similar companies.")
    st.markdown("**Try these tickers:** AAPL, TSLA, JPM, XOM, COST, BA")
