"""
ratio_engine.py — Compute common-size financial ratios (the "Financial DNA")

These 10 ratios normalize financial statements so companies of any size
can be compared on an apples-to-apples basis.
"""

import pandas as pd
import numpy as np

# The 10 common-size ratio definitions
RATIO_NAMES = [
    "COGS_to_Revenue",
    "Gross_Margin",
    "SGA_to_Revenue",
    "Net_Margin",
    "Cash_to_Assets",
    "Receivables_to_Assets",
    "Inventory_to_Assets",
    "PPE_to_Assets",
    "Debt_to_Assets",
    "Equity_to_Assets",
]

RATIO_LABELS = [
    "COGS / Revenue",
    "Gross Margin",
    "SG&A / Revenue",
    "Net Margin",
    "Cash / Assets",
    "Receivables / Assets",
    "Inventory / Assets",
    "PP&E / Assets",
    "Debt / Assets",
    "Equity / Assets",
]


def compute_ratios_from_yfinance(ticker_obj) -> dict:
    """
    Given a yfinance Ticker object, compute 10 common-size ratios
    from the most recent annual financial statements.

    Returns a dict of ratio values, or None if data is insufficient.
    """
    try:
        # Get annual financials
        income_stmt = ticker_obj.financials  # columns = dates, rows = line items
        balance_sheet = ticker_obj.balance_sheet

        if income_stmt.empty or balance_sheet.empty:
            return None

        # Use most recent year (first column)
        inc = income_stmt.iloc[:, 0]
        bal = balance_sheet.iloc[:, 0]

        # Helper to safely get a value from the series
        def get(series, keys, default=np.nan):
            if isinstance(keys, str):
                keys = [keys]
            for key in keys:
                if key in series.index and pd.notna(series[key]):
                    return float(series[key])
            return default

        # Income statement items
        revenue = get(inc, ["Total Revenue", "Revenue", "Operating Revenue"])
        cogs = get(inc, ["Cost Of Revenue", "Cost Of Goods Sold", "Cost of Revenue"])
        gross_profit = get(inc, ["Gross Profit"])
        sga = get(inc, ["Selling General And Administration",
                        "Selling General And Administrative",
                        "Operating Expense"])
        net_income = get(inc, ["Net Income", "Net Income Common Stockholders"])

        # Balance sheet items
        total_assets = get(bal, ["Total Assets"])
        cash = get(bal, ["Cash And Cash Equivalents",
                         "Cash Cash Equivalents And Short Term Investments",
                         "Cash Financial"])
        receivables = get(bal, ["Net Receivables", "Receivables",
                                "Accounts Receivable"])
        inventory = get(bal, ["Inventory"])
        ppe = get(bal, ["Net PPE", "Property Plant And Equipment Net",
                        "Property Plant Equipment Net",
                        "Gross PPE"])
        total_debt = get(bal, ["Total Debt",
                               "Total Liabilities Net Minority Interest",
                               "Current Debt And Capital Lease Obligation"])
        equity = get(bal, ["Total Equity Gross Minority Interest",
                           "Stockholders Equity",
                           "Common Stock Equity"])

        # Guard: need at least revenue and total assets
        if np.isnan(revenue) or revenue == 0 or np.isnan(total_assets) or total_assets == 0:
            return None

        ratios = {
            "COGS_to_Revenue": cogs / revenue if not np.isnan(cogs) else np.nan,
            "Gross_Margin": gross_profit / revenue if not np.isnan(gross_profit) else np.nan,
            "SGA_to_Revenue": sga / revenue if not np.isnan(sga) else np.nan,
            "Net_Margin": net_income / revenue if not np.isnan(net_income) else np.nan,
            "Cash_to_Assets": cash / total_assets if not np.isnan(cash) else np.nan,
            "Receivables_to_Assets": receivables / total_assets if not np.isnan(receivables) else np.nan,
            "Inventory_to_Assets": inventory / total_assets if not np.isnan(inventory) else np.nan,
            "PPE_to_Assets": ppe / total_assets if not np.isnan(ppe) else np.nan,
            "Debt_to_Assets": total_debt / total_assets if not np.isnan(total_debt) else np.nan,
            "Equity_to_Assets": equity / total_assets if not np.isnan(equity) else np.nan,
        }

        return ratios

    except Exception:
        return None


def compute_ratios_from_raw(inc_dict: dict, bal_dict: dict) -> dict:
    """
    Compute ratios from raw dictionaries of financial data.
    Useful for processing bulk-downloaded data.
    """
    def safe_get(d, keys, default=np.nan):
        if isinstance(keys, str):
            keys = [keys]
        for k in keys:
            if k in d and pd.notna(d[k]):
                return float(d[k])
        return default

    revenue = safe_get(inc_dict, ["Total Revenue", "Revenue", "Operating Revenue"])
    cogs = safe_get(inc_dict, ["Cost Of Revenue", "Cost Of Goods Sold"])
    gross_profit = safe_get(inc_dict, ["Gross Profit"])
    sga = safe_get(inc_dict, ["Selling General And Administration",
                               "Selling General And Administrative",
                               "Operating Expense"])
    net_income = safe_get(inc_dict, ["Net Income", "Net Income Common Stockholders"])

    total_assets = safe_get(bal_dict, ["Total Assets"])
    cash = safe_get(bal_dict, ["Cash And Cash Equivalents",
                               "Cash Cash Equivalents And Short Term Investments"])
    receivables = safe_get(bal_dict, ["Net Receivables", "Receivables",
                                      "Accounts Receivable"])
    inventory = safe_get(bal_dict, ["Inventory"])
    ppe = safe_get(bal_dict, ["Net PPE", "Property Plant And Equipment Net",
                              "Gross PPE"])
    total_debt = safe_get(bal_dict, ["Total Debt",
                                     "Total Liabilities Net Minority Interest"])
    equity = safe_get(bal_dict, ["Total Equity Gross Minority Interest",
                                 "Stockholders Equity",
                                 "Common Stock Equity"])

    if np.isnan(revenue) or revenue == 0 or np.isnan(total_assets) or total_assets == 0:
        return None

    return {
        "COGS_to_Revenue": cogs / revenue if not np.isnan(cogs) else np.nan,
        "Gross_Margin": gross_profit / revenue if not np.isnan(gross_profit) else np.nan,
        "SGA_to_Revenue": sga / revenue if not np.isnan(sga) else np.nan,
        "Net_Margin": net_income / revenue if not np.isnan(net_income) else np.nan,
        "Cash_to_Assets": cash / total_assets if not np.isnan(cash) else np.nan,
        "Receivables_to_Assets": receivables / total_assets if not np.isnan(receivables) else np.nan,
        "Inventory_to_Assets": inventory / total_assets if not np.isnan(inventory) else np.nan,
        "PPE_to_Assets": ppe / total_assets if not np.isnan(ppe) else np.nan,
        "Debt_to_Assets": total_debt / total_assets if not np.isnan(total_debt) else np.nan,
        "Equity_to_Assets": equity / total_assets if not np.isnan(equity) else np.nan,
    }


def ratios_to_vector(ratios: dict) -> list:
    """Convert a ratio dict to a feature vector in the standard order."""
    return [ratios.get(name, np.nan) for name in RATIO_NAMES]
