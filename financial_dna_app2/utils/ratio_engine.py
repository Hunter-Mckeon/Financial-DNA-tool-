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


def _ratios_from_series(inc, bal) -> dict:
    """
    Internal helper: compute the 10 ratios from a pair of pandas Series
    (one fiscal year's income-statement column and balance-sheet column).
    Returns None if revenue or total assets are missing / zero.
    """
    def get(series, keys, default=np.nan):
        if isinstance(keys, str):
            keys = [keys]
        for key in keys:
            if key in series.index and pd.notna(series[key]):
                return float(series[key])
        return default

    # Income statement
    revenue = get(inc, ["Total Revenue", "Revenue", "Operating Revenue"])
    cogs = get(inc, ["Cost Of Revenue", "Cost Of Goods Sold", "Cost of Revenue"])
    gross_profit = get(inc, ["Gross Profit"])
    # Do NOT fall back to "Operating Expense" — it includes R&D and other
    # non-SG&A items, which inflates the ratio.
    sga = get(inc, ["Selling General And Administration",
                    "Selling General And Administrative"])
    net_income = get(inc, ["Net Income", "Net Income Common Stockholders"])
    if np.isnan(gross_profit) and not np.isnan(cogs) and not np.isnan(revenue):
        gross_profit = revenue - cogs

    # Balance sheet
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
    # "Total Liabilities" != "Total Debt" — prefer explicit debt lines.
    total_debt = get(bal, ["Total Debt"])
    if np.isnan(total_debt):
        ltd = get(bal, ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"])
        cd = get(bal, ["Current Debt", "Current Debt And Capital Lease Obligation",
                       "Short Long Term Debt"])
        if not np.isnan(ltd) or not np.isnan(cd):
            total_debt = (0 if np.isnan(ltd) else ltd) + (0 if np.isnan(cd) else cd)
    equity = get(bal, ["Total Equity Gross Minority Interest",
                       "Stockholders Equity",
                       "Common Stock Equity"])

    # Guard: need revenue and total assets to compute anything meaningful
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


def compute_ratios_from_yfinance(ticker_obj) -> dict:
    """
    Given a yfinance Ticker object, compute 10 common-size ratios
    from the most recent annual financial statements.

    Returns a dict of ratio values, or None if data is insufficient.
    """
    try:
        income_stmt = ticker_obj.financials  # columns = dates, rows = line items
        balance_sheet = ticker_obj.balance_sheet

        if income_stmt.empty or balance_sheet.empty:
            return None

        # Most recent year only (first column)
        return _ratios_from_series(income_stmt.iloc[:, 0], balance_sheet.iloc[:, 0])

    except Exception:
        return None


def compute_ratios_multi_year(ticker_obj, max_years: int = 5) -> list:
    """
    Given a yfinance Ticker object, compute ratios for up to `max_years` of
    historical annual financial statements, returning a list of
    (fiscal_year_end_date_str, ratios_dict) tuples — newest first.

    This is the workhorse for building multi-year panel data: each fiscal year
    of each company becomes one training row.  Years with insufficient data
    are silently skipped so one bad year doesn't drop the whole company.
    """
    out = []
    try:
        income_stmt = ticker_obj.financials
        balance_sheet = ticker_obj.balance_sheet

        if income_stmt.empty or balance_sheet.empty:
            return out

        # Intersect the fiscal year columns present in both statements
        common_dates = [d for d in income_stmt.columns if d in balance_sheet.columns]
        # yfinance orders newest-first, but make it explicit
        common_dates = sorted(common_dates, reverse=True)[:max_years]

        for date_col in common_dates:
            try:
                inc = income_stmt[date_col]
                bal = balance_sheet[date_col]
                r = _ratios_from_series(inc, bal)
                if r is not None:
                    # Stringify the fiscal-year-end for the CSV
                    year_label = pd.Timestamp(date_col).strftime("%Y-%m-%d")
                    out.append((year_label, r))
            except Exception:
                continue

    except Exception:
        pass
    return out


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
    # Do NOT fall back to "Operating Expense" — it includes R&D and other non-SG&A items.
    sga = safe_get(inc_dict, ["Selling General And Administration",
                               "Selling General And Administrative"])
    net_income = safe_get(inc_dict, ["Net Income", "Net Income Common Stockholders"])

    if np.isnan(gross_profit) and not np.isnan(cogs) and not np.isnan(revenue):
        gross_profit = revenue - cogs

    total_assets = safe_get(bal_dict, ["Total Assets"])
    cash = safe_get(bal_dict, ["Cash And Cash Equivalents",
                               "Cash Cash Equivalents And Short Term Investments"])
    receivables = safe_get(bal_dict, ["Net Receivables", "Receivables",
                                      "Accounts Receivable"])
    inventory = safe_get(bal_dict, ["Inventory"])
    ppe = safe_get(bal_dict, ["Net PPE", "Property Plant And Equipment Net",
                              "Gross PPE"])
    # "Total Liabilities" != "Total Debt" — prefer explicit debt line items.
    total_debt = safe_get(bal_dict, ["Total Debt"])
    if np.isnan(total_debt):
        ltd = safe_get(bal_dict, ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"])
        cd = safe_get(bal_dict, ["Current Debt", "Current Debt And Capital Lease Obligation",
                                 "Short Long Term Debt"])
        if not np.isnan(ltd) or not np.isnan(cd):
            total_debt = (0 if np.isnan(ltd) else ltd) + (0 if np.isnan(cd) else cd)
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
