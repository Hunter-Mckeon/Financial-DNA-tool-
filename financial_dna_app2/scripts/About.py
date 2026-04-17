"""
Financial DNA — Decode Any Company's Financial Fingerprint
BA870/AC820 Team Project | Spring 2026

Main entry point / Home page for the Streamlit app.
"""

import streamlit as st
import pandas as pd
import os

st.set_page_config(
    page_title="Financial DNA",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #6c757d;
        text-align: center;
        margin-top: 0;
        margin-bottom: 2rem;
    }
    /* Use a subtle translucent background so metric cards work in both
       light and dark themes.  rgba(128,128,128,0.08) reads as a faint
       gray overlay on either. */
    [data-testid="stMetric"] {
        background-color: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.15);
        border-radius: 10px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──
st.sidebar.markdown("## 🧬 Financial DNA")
st.sidebar.markdown("---")
st.sidebar.markdown("""
**Navigate the app:**
- **Industry Classifier** — Predict a company's industry from its financials
- **Peer Finder** — Find the most similar companies
- **Company Comparison** — Compare two companies head-to-head
- **Model Performance** — View ML model accuracy & explainability
""")
st.sidebar.markdown("---")
st.sidebar.markdown("""
**Team:**
Howard Mckeon
Joshua Hartono
Adam Schuler

**Course:** BA870/AC820
**Spring 2026**
""")

# ── Main Content ──
st.markdown('<p class="main-header">🧬 Financial DNA</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Decode Any Company\'s Financial Fingerprint</p>', unsafe_allow_html=True)

st.markdown("---")

# Intro section
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ### What is Financial DNA?

    Every company has a unique **financial fingerprint** — a set of common-size ratios
    derived from its financial statements that reveal its cost structure, asset composition,
    and profitability profile.

    This app uses **machine learning** to analyze these ratios and:

    1. **Classify** a company's industry based purely on its financial structure
    2. **Find peers** — companies with the most similar financial profiles
    3. **Compare** any two companies side-by-side with interactive radar charts
    4. **Explain** which financial ratios matter most for classification

    Enter any stock ticker and let the models decode its Financial DNA.
    """)

with col2:
    st.markdown("### The 10 Ratio Features")
    st.markdown("""
    | Ratio | Category |
    |-------|----------|
    | COGS / Revenue | Cost |
    | Gross Margin | Profitability |
    | SG&A / Revenue | Overhead |
    | Net Margin | Profitability |
    | Cash / Assets | Liquidity |
    | Receivables / Assets | Working Capital |
    | Inventory / Assets | Working Capital |
    | PP&E / Assets | Asset Mix |
    | Debt / Assets | Leverage |
    | Equity / Assets | Capitalization |
    """)

st.markdown("---")

# Load training data stats
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
try:
    df = pd.read_csv(os.path.join(DATA_DIR, "company_ratios.csv"))

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    # Count unique tickers (multiple fiscal years per ticker => more rows than companies)
    n_companies = df["ticker"].nunique() if "ticker" in df.columns else len(df)
    n_rows = len(df)

    with col1:
        st.metric("Companies in Database", f"{n_companies}")
    with col2:
        st.metric("Industry Sectors", f"{df['sector'].nunique()}")
    with col3:
        st.metric("Sub-Industries", f"{df['industry'].nunique()}")
    with col4:
        # Load model metrics if available
        try:
            import pickle
            MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
            with open(os.path.join(MODEL_DIR, "metrics.pkl"), "rb") as f:
                metrics = pickle.load(f)
            best_acc = max(m["accuracy"] for m in metrics.values())
            st.metric("Best Model Accuracy", f"{best_acc:.1%}")
        except Exception:
            st.metric("ML Models", "4")

    # Small caption below the metrics row with the fiscal-year detail
    if n_rows > n_companies:
        st.caption(f"Trained on {n_rows} company-year observations ({n_rows // n_companies}-ish fiscal years per company on average)")

    st.markdown("---")

    # Sector distribution chart
    st.markdown("### Training Data: Sector Distribution")
    sector_counts = df["sector"].value_counts().reset_index()
    sector_counts.columns = ["Sector", "Count"]
    st.bar_chart(sector_counts.set_index("Sector"))

except FileNotFoundError:
    st.warning("Training data not found. Please run `python scripts/build_dataset.py` first.")

st.markdown("---")

# How it works section
st.markdown("""
### How It Works

**Data Pipeline:** The app fetches a company's annual income statement and balance sheet
via yfinance, then computes 10 common-size ratios that normalize the financial data
regardless of company size.

**Machine Learning Models:**
- **Random Forest** — Primary classifier using ensemble of decision trees
- **XGBoost** — Gradient-boosted trees for comparison
- **Logistic Regression** — Simple baseline to benchmark against
- **K-Nearest Neighbors** — Used for peer identification via distance in ratio-space

**Get started** by selecting a page from the sidebar!
""")
