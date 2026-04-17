"""
data_fetcher.py — Fetch financial data from yfinance with caching.
"""

import time
import yfinance as yf
import pandas as pd
import numpy as np
from functools import lru_cache
from utils.ratio_engine import (
    compute_ratios_from_yfinance, compute_ratios_multi_year, RATIO_NAMES,
)


# Yahoo Finance returns sectors in its own taxonomy; we normalize to GICS so they
# match the training data.  Anything not in this map passes through unchanged.
YF_TO_GICS_SECTOR = {
    "Financial Services": "Financials",
    "Healthcare": "Health Care",
    "Consumer Cyclical": "Consumer Discretionary",
    "Consumer Defensive": "Consumer Staples",
    "Basic Materials": "Materials",
    # These already match GICS:
    # Technology, Industrials, Energy, Utilities, Communication Services, Real Estate
}


def normalize_sector(sector_name: str) -> str:
    """Map a Yahoo-style sector name to its GICS equivalent."""
    if not sector_name:
        return "Unknown"
    return YF_TO_GICS_SECTOR.get(sector_name, sector_name)


def get_sp500_tickers() -> pd.DataFrame:
    """
    Get S&P 500 tickers from Wikipedia.
    Returns a DataFrame with 'Symbol', 'Security', 'GICS Sector', 'GICS Sub-Industry'.
    """
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        df = tables[0]
        # Clean up ticker symbols (some have dots like BRK.B)
        df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)
        return df[["Symbol", "Security", "GICS Sector", "GICS Sub-Industry"]]
    except Exception:
        # Fallback: return a hardcoded list of major tickers
        return _fallback_tickers()


def _fallback_tickers() -> pd.DataFrame:
    """Hardcoded fallback list of ~100 well-known tickers."""
    tickers = [
        ("AAPL", "Apple Inc", "Information Technology", "Technology Hardware"),
        ("MSFT", "Microsoft Corp", "Information Technology", "Systems Software"),
        ("AMZN", "Amazon.com Inc", "Consumer Discretionary", "Internet Retail"),
        ("GOOGL", "Alphabet Inc", "Communication Services", "Interactive Media"),
        ("META", "Meta Platforms", "Communication Services", "Interactive Media"),
        ("TSLA", "Tesla Inc", "Consumer Discretionary", "Auto Manufacturers"),
        ("NVDA", "NVIDIA Corp", "Information Technology", "Semiconductors"),
        ("JPM", "JPMorgan Chase", "Financials", "Diversified Banks"),
        ("JNJ", "Johnson & Johnson", "Health Care", "Pharmaceuticals"),
        ("V", "Visa Inc", "Financials", "Transaction Processing"),
        ("PG", "Procter & Gamble", "Consumer Staples", "Household Products"),
        ("UNH", "UnitedHealth Group", "Health Care", "Managed Health Care"),
        ("HD", "Home Depot", "Consumer Discretionary", "Home Improvement"),
        ("MA", "Mastercard", "Financials", "Transaction Processing"),
        ("DIS", "Walt Disney", "Communication Services", "Movies & Entertainment"),
        ("BAC", "Bank of America", "Financials", "Diversified Banks"),
        ("XOM", "Exxon Mobil", "Energy", "Integrated Oil & Gas"),
        ("PFE", "Pfizer Inc", "Health Care", "Pharmaceuticals"),
        ("KO", "Coca-Cola Co", "Consumer Staples", "Soft Drinks"),
        ("PEP", "PepsiCo Inc", "Consumer Staples", "Soft Drinks"),
        ("CSCO", "Cisco Systems", "Information Technology", "Communications Equipment"),
        ("ABBV", "AbbVie Inc", "Health Care", "Pharmaceuticals"),
        ("COST", "Costco Wholesale", "Consumer Staples", "Hypermarkets"),
        ("TMO", "Thermo Fisher", "Health Care", "Life Sciences Tools"),
        ("AVGO", "Broadcom Inc", "Information Technology", "Semiconductors"),
        ("CVX", "Chevron Corp", "Energy", "Integrated Oil & Gas"),
        ("WMT", "Walmart Inc", "Consumer Staples", "Hypermarkets"),
        ("MRK", "Merck & Co", "Health Care", "Pharmaceuticals"),
        ("ABT", "Abbott Labs", "Health Care", "Health Care Equipment"),
        ("LLY", "Eli Lilly", "Health Care", "Pharmaceuticals"),
        ("CRM", "Salesforce Inc", "Information Technology", "Application Software"),
        ("ADBE", "Adobe Inc", "Information Technology", "Application Software"),
        ("ORCL", "Oracle Corp", "Information Technology", "Application Software"),
        ("INTC", "Intel Corp", "Information Technology", "Semiconductors"),
        ("AMD", "AMD Inc", "Information Technology", "Semiconductors"),
        ("NFLX", "Netflix Inc", "Communication Services", "Movies & Entertainment"),
        ("NKE", "Nike Inc", "Consumer Discretionary", "Footwear"),
        ("MCD", "McDonalds Corp", "Consumer Discretionary", "Restaurants"),
        ("LOW", "Lowes Companies", "Consumer Discretionary", "Home Improvement"),
        ("GS", "Goldman Sachs", "Financials", "Investment Banking"),
        ("CAT", "Caterpillar Inc", "Industrials", "Construction Machinery"),
        ("BA", "Boeing Co", "Industrials", "Aerospace & Defense"),
        ("RTX", "RTX Corp", "Industrials", "Aerospace & Defense"),
        ("GE", "GE Aerospace", "Industrials", "Aerospace & Defense"),
        ("HON", "Honeywell", "Industrials", "Industrial Conglomerates"),
        ("UPS", "United Parcel Service", "Industrials", "Air Freight & Logistics"),
        ("MMM", "3M Company", "Industrials", "Industrial Conglomerates"),
        ("DE", "Deere & Co", "Industrials", "Farm Machinery"),
        ("NEE", "NextEra Energy", "Utilities", "Electric Utilities"),
        ("DUK", "Duke Energy", "Utilities", "Electric Utilities"),
        ("SO", "Southern Company", "Utilities", "Electric Utilities"),
        ("D", "Dominion Energy", "Utilities", "Electric Utilities"),
        ("AEP", "American Electric Power", "Utilities", "Electric Utilities"),
        ("SLB", "Schlumberger", "Energy", "Oil & Gas Equipment"),
        ("COP", "ConocoPhillips", "Energy", "Oil & Gas Exploration"),
        ("EOG", "EOG Resources", "Energy", "Oil & Gas Exploration"),
        ("PSX", "Phillips 66", "Energy", "Oil & Gas Refining"),
        ("VLO", "Valero Energy", "Energy", "Oil & Gas Refining"),
        ("AMT", "American Tower", "Real Estate", "Telecom Tower REITs"),
        ("PLD", "Prologis Inc", "Real Estate", "Industrial REITs"),
        ("SPG", "Simon Property Group", "Real Estate", "Retail REITs"),
        ("LIN", "Linde PLC", "Materials", "Industrial Gases"),
        ("APD", "Air Products", "Materials", "Industrial Gases"),
        ("SHW", "Sherwin-Williams", "Materials", "Specialty Chemicals"),
        ("FCX", "Freeport-McMoRan", "Materials", "Copper"),
        ("NEM", "Newmont Corp", "Materials", "Gold"),
        ("WFC", "Wells Fargo", "Financials", "Diversified Banks"),
        ("C", "Citigroup Inc", "Financials", "Diversified Banks"),
        ("MS", "Morgan Stanley", "Financials", "Investment Banking"),
        ("BLK", "BlackRock Inc", "Financials", "Asset Management"),
        ("SCHW", "Charles Schwab", "Financials", "Investment Banking"),
        ("AXP", "American Express", "Financials", "Consumer Finance"),
    ]
    return pd.DataFrame(tickers, columns=["Symbol", "Security", "GICS Sector", "GICS Sub-Industry"])


def fetch_company_data(ticker: str,
                        preferred_sector: str = None,
                        preferred_industry: str = None,
                        preferred_name: str = None,
                        max_retries: int = 3) -> dict:
    """
    Fetch a single company's info, ratios, and classification from yfinance,
    with retry/backoff for transient rate-limit errors.

    If `preferred_sector` / `preferred_industry` are provided (e.g. the GICS
    Sector column scraped from Wikipedia) they override yfinance's Yahoo-style
    sector labels so training data uses a single consistent taxonomy.
    """
    last_err = None
    for attempt in range(max_retries):
        try:
            t = yf.Ticker(ticker)
            ratios = compute_ratios_from_yfinance(t)
            if ratios is None:
                return None

            # Pull info for name/fallback sector; tolerate rate limits on this call.
            try:
                info = t.info or {}
            except Exception:
                info = {}

            sector = preferred_sector or normalize_sector(info.get("sector", "Unknown"))
            industry = preferred_industry or info.get("industry", "Unknown")
            name = preferred_name or info.get("longName", info.get("shortName", ticker))

            return {
                "ticker": ticker,
                "company_name": name,
                "sector": sector,
                "industry": industry,
                **ratios,
            }
        except Exception as e:
            last_err = e
            # Exponential backoff: 1s, 2s, 4s
            time.sleep(2 ** attempt)
    return None


def fetch_company_data_multi_year(ticker: str,
                                   preferred_sector: str = None,
                                   preferred_industry: str = None,
                                   preferred_name: str = None,
                                   max_years: int = 5,
                                   max_retries: int = 3) -> list:
    """
    Fetch up to `max_years` of historical annual ratios for one ticker.

    Returns a list of row-dicts (one per fiscal year), or an empty list if the
    ticker fails.  The ticker field is suffixed with the fiscal-year-end
    (e.g. "AAPL@2023-09-30") so each row is unique in the training CSV, while
    the `company_name`, `sector`, and `industry` stay constant across years.
    """
    last_err = None
    for attempt in range(max_retries):
        try:
            t = yf.Ticker(ticker)
            yearly = compute_ratios_multi_year(t, max_years=max_years)
            if not yearly:
                return []

            # Resolve sector/industry/name once per ticker (info call is the
            # rate-limited one, so do it only here).
            try:
                info = t.info or {}
            except Exception:
                info = {}

            sector = preferred_sector or normalize_sector(info.get("sector", "Unknown"))
            industry = preferred_industry or info.get("industry", "Unknown")
            name = preferred_name or info.get("longName", info.get("shortName", ticker))

            rows = []
            for year_label, ratios in yearly:
                # Keep `ticker` as the plain symbol so existing page lookups
                # (training_df[training_df["ticker"] == ticker].iloc[0]) still
                # work — iloc[0] naturally returns the most recent year since
                # we iterate newest-first.  `fiscal_year_end` distinguishes
                # the rows within the CSV.
                rows.append({
                    "ticker": ticker,
                    "company_name": name,
                    "sector": sector,
                    "industry": industry,
                    "fiscal_year_end": year_label,
                    **ratios,
                })
            return rows
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
    return []


@lru_cache(maxsize=256)
def fetch_company_ratios_live(ticker: str) -> dict:
    """
    Fetch ratios for a single ticker in real-time (for app predictions).
    Returns ratio dict or None.

    Cached per-process so repeated lookups of the same ticker during a Streamlit
    session don't re-hit the Yahoo Finance API.  (Clear with
    `fetch_company_ratios_live.cache_clear()` if needed.)

    The returned `sector` is normalized to GICS so it matches the training data.
    """
    try:
        t = yf.Ticker(ticker)
        ratios = compute_ratios_from_yfinance(t)
        if ratios is None:
            return None
        info = t.info or {}
        return {
            "ticker": ticker,
            "company_name": info.get("longName", info.get("shortName", ticker)),
            "sector": normalize_sector(info.get("sector", "Unknown")),
            "industry": info.get("industry", "Unknown"),
            **ratios,
        }
    except Exception:
        return None
