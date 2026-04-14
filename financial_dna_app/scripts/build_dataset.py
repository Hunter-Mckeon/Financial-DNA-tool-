"""
build_dataset.py — Build the training dataset for Financial DNA.

Two modes:
  1. LIVE MODE (default when yfinance works): Pull from yfinance for S&P 500 companies.
  2. SYNTHETIC MODE (fallback): Generate realistic data based on known industry profiles.

Run: python scripts/build_dataset.py [--synthetic]
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import time

from utils.ratio_engine import RATIO_NAMES

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


# ── Realistic industry ratio profiles (mean, std) based on public financial data ──
# Each sector has characteristic financial "DNA" patterns
SECTOR_PROFILES = {
    "Technology": {
        "COGS_to_Revenue":       (0.35, 0.15),
        "Gross_Margin":          (0.65, 0.15),
        "SGA_to_Revenue":        (0.30, 0.12),
        "Net_Margin":            (0.18, 0.12),
        "Cash_to_Assets":        (0.25, 0.12),
        "Receivables_to_Assets": (0.15, 0.08),
        "Inventory_to_Assets":   (0.03, 0.04),
        "PPE_to_Assets":         (0.10, 0.08),
        "Debt_to_Assets":        (0.55, 0.15),
        "Equity_to_Assets":      (0.40, 0.15),
    },
    "Health Care": {
        "COGS_to_Revenue":       (0.40, 0.18),
        "Gross_Margin":          (0.60, 0.18),
        "SGA_to_Revenue":        (0.28, 0.10),
        "Net_Margin":            (0.12, 0.15),
        "Cash_to_Assets":        (0.18, 0.12),
        "Receivables_to_Assets": (0.12, 0.07),
        "Inventory_to_Assets":   (0.08, 0.06),
        "PPE_to_Assets":         (0.12, 0.08),
        "Debt_to_Assets":        (0.58, 0.15),
        "Equity_to_Assets":      (0.38, 0.15),
    },
    "Financials": {
        "COGS_to_Revenue":       (0.15, 0.12),
        "Gross_Margin":          (0.85, 0.12),
        "SGA_to_Revenue":        (0.45, 0.15),
        "Net_Margin":            (0.22, 0.12),
        "Cash_to_Assets":        (0.08, 0.06),
        "Receivables_to_Assets": (0.05, 0.04),
        "Inventory_to_Assets":   (0.01, 0.02),
        "PPE_to_Assets":         (0.03, 0.03),
        "Debt_to_Assets":        (0.85, 0.08),
        "Equity_to_Assets":      (0.12, 0.06),
    },
    "Consumer Discretionary": {
        "COGS_to_Revenue":       (0.60, 0.15),
        "Gross_Margin":          (0.40, 0.15),
        "SGA_to_Revenue":        (0.22, 0.10),
        "Net_Margin":            (0.06, 0.08),
        "Cash_to_Assets":        (0.10, 0.08),
        "Receivables_to_Assets": (0.08, 0.06),
        "Inventory_to_Assets":   (0.15, 0.10),
        "PPE_to_Assets":         (0.20, 0.12),
        "Debt_to_Assets":        (0.60, 0.12),
        "Equity_to_Assets":      (0.35, 0.12),
    },
    "Consumer Staples": {
        "COGS_to_Revenue":       (0.55, 0.12),
        "Gross_Margin":          (0.45, 0.12),
        "SGA_to_Revenue":        (0.20, 0.08),
        "Net_Margin":            (0.08, 0.06),
        "Cash_to_Assets":        (0.06, 0.05),
        "Receivables_to_Assets": (0.08, 0.05),
        "Inventory_to_Assets":   (0.12, 0.08),
        "PPE_to_Assets":         (0.22, 0.10),
        "Debt_to_Assets":        (0.62, 0.10),
        "Equity_to_Assets":      (0.33, 0.10),
    },
    "Energy": {
        "COGS_to_Revenue":       (0.70, 0.12),
        "Gross_Margin":          (0.30, 0.12),
        "SGA_to_Revenue":        (0.08, 0.05),
        "Net_Margin":            (0.10, 0.12),
        "Cash_to_Assets":        (0.05, 0.04),
        "Receivables_to_Assets": (0.10, 0.06),
        "Inventory_to_Assets":   (0.05, 0.04),
        "PPE_to_Assets":         (0.50, 0.15),
        "Debt_to_Assets":        (0.55, 0.12),
        "Equity_to_Assets":      (0.42, 0.12),
    },
    "Industrials": {
        "COGS_to_Revenue":       (0.65, 0.12),
        "Gross_Margin":          (0.35, 0.12),
        "SGA_to_Revenue":        (0.15, 0.08),
        "Net_Margin":            (0.08, 0.06),
        "Cash_to_Assets":        (0.08, 0.06),
        "Receivables_to_Assets": (0.15, 0.07),
        "Inventory_to_Assets":   (0.12, 0.08),
        "PPE_to_Assets":         (0.25, 0.12),
        "Debt_to_Assets":        (0.60, 0.10),
        "Equity_to_Assets":      (0.35, 0.10),
    },
    "Utilities": {
        "COGS_to_Revenue":       (0.55, 0.15),
        "Gross_Margin":          (0.45, 0.15),
        "SGA_to_Revenue":        (0.10, 0.05),
        "Net_Margin":            (0.12, 0.06),
        "Cash_to_Assets":        (0.02, 0.02),
        "Receivables_to_Assets": (0.06, 0.04),
        "Inventory_to_Assets":   (0.03, 0.03),
        "PPE_to_Assets":         (0.55, 0.12),
        "Debt_to_Assets":        (0.70, 0.08),
        "Equity_to_Assets":      (0.28, 0.08),
    },
    "Materials": {
        "COGS_to_Revenue":       (0.68, 0.10),
        "Gross_Margin":          (0.32, 0.10),
        "SGA_to_Revenue":        (0.10, 0.05),
        "Net_Margin":            (0.08, 0.08),
        "Cash_to_Assets":        (0.05, 0.04),
        "Receivables_to_Assets": (0.12, 0.06),
        "Inventory_to_Assets":   (0.12, 0.07),
        "PPE_to_Assets":         (0.35, 0.12),
        "Debt_to_Assets":        (0.55, 0.10),
        "Equity_to_Assets":      (0.42, 0.10),
    },
    "Real Estate": {
        "COGS_to_Revenue":       (0.30, 0.18),
        "Gross_Margin":          (0.70, 0.18),
        "SGA_to_Revenue":        (0.12, 0.08),
        "Net_Margin":            (0.20, 0.15),
        "Cash_to_Assets":        (0.03, 0.03),
        "Receivables_to_Assets": (0.04, 0.03),
        "Inventory_to_Assets":   (0.02, 0.03),
        "PPE_to_Assets":         (0.60, 0.15),
        "Debt_to_Assets":        (0.65, 0.10),
        "Equity_to_Assets":      (0.32, 0.10),
    },
    "Communication Services": {
        "COGS_to_Revenue":       (0.42, 0.15),
        "Gross_Margin":          (0.58, 0.15),
        "SGA_to_Revenue":        (0.25, 0.10),
        "Net_Margin":            (0.15, 0.12),
        "Cash_to_Assets":        (0.12, 0.08),
        "Receivables_to_Assets": (0.12, 0.06),
        "Inventory_to_Assets":   (0.02, 0.03),
        "PPE_to_Assets":         (0.18, 0.12),
        "Debt_to_Assets":        (0.58, 0.12),
        "Equity_to_Assets":      (0.38, 0.12),
    },
}

# Representative companies per sector (ticker, name, sub-industry)
SECTOR_COMPANIES = {
    "Technology": [
        ("AAPL", "Apple Inc", "Technology Hardware"), ("MSFT", "Microsoft Corp", "Systems Software"),
        ("NVDA", "NVIDIA Corp", "Semiconductors"), ("AVGO", "Broadcom Inc", "Semiconductors"),
        ("ADBE", "Adobe Inc", "Application Software"), ("CRM", "Salesforce Inc", "Application Software"),
        ("ORCL", "Oracle Corp", "Application Software"), ("CSCO", "Cisco Systems", "Communications Equipment"),
        ("INTC", "Intel Corp", "Semiconductors"), ("AMD", "AMD Inc", "Semiconductors"),
        ("TXN", "Texas Instruments", "Semiconductors"), ("QCOM", "Qualcomm Inc", "Semiconductors"),
        ("IBM", "IBM Corp", "IT Consulting"), ("NOW", "ServiceNow Inc", "Application Software"),
        ("INTU", "Intuit Inc", "Application Software"), ("AMAT", "Applied Materials", "Semiconductor Equipment"),
        ("LRCX", "Lam Research", "Semiconductor Equipment"), ("MU", "Micron Technology", "Semiconductors"),
        ("SNPS", "Synopsys Inc", "Application Software"), ("CDNS", "Cadence Design", "Application Software"),
        ("KLAC", "KLA Corp", "Semiconductor Equipment"), ("MCHP", "Microchip Technology", "Semiconductors"),
        ("MSI", "Motorola Solutions", "Communications Equipment"), ("HPQ", "HP Inc", "Technology Hardware"),
        ("DELL", "Dell Technologies", "Technology Hardware"), ("FTNT", "Fortinet Inc", "Systems Software"),
        ("PANW", "Palo Alto Networks", "Systems Software"), ("CRWD", "CrowdStrike Holdings", "Systems Software"),
        ("WDAY", "Workday Inc", "Application Software"), ("TEAM", "Atlassian Corp", "Application Software"),
        ("DDOG", "Datadog Inc", "Application Software"), ("ZS", "Zscaler Inc", "Systems Software"),
        ("HUBS", "HubSpot Inc", "Application Software"), ("DOCU", "DocuSign Inc", "Application Software"),
        ("SPLK", "Splunk Inc", "Application Software"), ("NET", "Cloudflare Inc", "Systems Software"),
        ("SNOW", "Snowflake Inc", "Application Software"), ("PLTR", "Palantir Technologies", "Application Software"),
        ("SHOP", "Shopify Inc", "Application Software"), ("SQ", "Block Inc", "Transaction Processing"),
    ],
    "Health Care": [
        ("JNJ", "Johnson & Johnson", "Pharmaceuticals"), ("UNH", "UnitedHealth Group", "Managed Health Care"),
        ("PFE", "Pfizer Inc", "Pharmaceuticals"), ("ABBV", "AbbVie Inc", "Pharmaceuticals"),
        ("TMO", "Thermo Fisher Scientific", "Life Sciences Tools"), ("MRK", "Merck & Co", "Pharmaceuticals"),
        ("ABT", "Abbott Laboratories", "Health Care Equipment"), ("LLY", "Eli Lilly", "Pharmaceuticals"),
        ("DHR", "Danaher Corp", "Life Sciences Tools"), ("BMY", "Bristol-Myers Squibb", "Pharmaceuticals"),
        ("AMGN", "Amgen Inc", "Biotechnology"), ("GILD", "Gilead Sciences", "Biotechnology"),
        ("MDT", "Medtronic PLC", "Health Care Equipment"), ("ISRG", "Intuitive Surgical", "Health Care Equipment"),
        ("CVS", "CVS Health Corp", "Health Care Services"), ("CI", "Cigna Group", "Managed Health Care"),
        ("SYK", "Stryker Corp", "Health Care Equipment"), ("BDX", "Becton Dickinson", "Health Care Equipment"),
        ("ZTS", "Zoetis Inc", "Pharmaceuticals"), ("VRTX", "Vertex Pharmaceuticals", "Biotechnology"),
        ("REGN", "Regeneron Pharmaceuticals", "Biotechnology"), ("BSX", "Boston Scientific", "Health Care Equipment"),
        ("ELV", "Elevance Health", "Managed Health Care"), ("HUM", "Humana Inc", "Managed Health Care"),
        ("MRNA", "Moderna Inc", "Biotechnology"), ("BIIB", "Biogen Inc", "Biotechnology"),
        ("A", "Agilent Technologies", "Life Sciences Tools"), ("IQV", "IQVIA Holdings", "Life Sciences Tools"),
        ("HOLX", "Hologic Inc", "Health Care Equipment"), ("DXCM", "DexCom Inc", "Health Care Equipment"),
    ],
    "Financials": [
        ("JPM", "JPMorgan Chase", "Diversified Banks"), ("V", "Visa Inc", "Transaction Processing"),
        ("MA", "Mastercard Inc", "Transaction Processing"), ("BAC", "Bank of America", "Diversified Banks"),
        ("WFC", "Wells Fargo", "Diversified Banks"), ("GS", "Goldman Sachs", "Investment Banking"),
        ("MS", "Morgan Stanley", "Investment Banking"), ("BLK", "BlackRock Inc", "Asset Management"),
        ("SCHW", "Charles Schwab", "Investment Banking"), ("AXP", "American Express", "Consumer Finance"),
        ("C", "Citigroup Inc", "Diversified Banks"), ("USB", "US Bancorp", "Regional Banks"),
        ("PNC", "PNC Financial", "Regional Banks"), ("TFC", "Truist Financial", "Regional Banks"),
        ("COF", "Capital One Financial", "Consumer Finance"), ("BK", "Bank of New York Mellon", "Asset Management"),
        ("ICE", "Intercontinental Exchange", "Financial Exchanges"), ("CME", "CME Group", "Financial Exchanges"),
        ("MCO", "Moodys Corp", "Financial Exchanges"), ("SPGI", "S&P Global", "Financial Exchanges"),
        ("MMC", "Marsh McLennan", "Insurance Brokers"), ("AON", "Aon PLC", "Insurance Brokers"),
        ("MET", "MetLife Inc", "Life Insurance"), ("PRU", "Prudential Financial", "Life Insurance"),
        ("AIG", "American International Group", "Multi-line Insurance"), ("ALL", "Allstate Corp", "Property Insurance"),
        ("TRV", "Travelers Companies", "Property Insurance"), ("CB", "Chubb Limited", "Property Insurance"),
        ("PGR", "Progressive Corp", "Property Insurance"), ("AFL", "Aflac Inc", "Life Insurance"),
    ],
    "Consumer Discretionary": [
        ("AMZN", "Amazon.com Inc", "Internet Retail"), ("TSLA", "Tesla Inc", "Auto Manufacturers"),
        ("HD", "Home Depot", "Home Improvement"), ("MCD", "McDonalds Corp", "Restaurants"),
        ("LOW", "Lowes Companies", "Home Improvement"), ("NKE", "Nike Inc", "Footwear"),
        ("SBUX", "Starbucks Corp", "Restaurants"), ("TJX", "TJX Companies", "Apparel Retail"),
        ("BKNG", "Booking Holdings", "Internet Travel"), ("CMG", "Chipotle Mexican Grill", "Restaurants"),
        ("ORLY", "OReilly Automotive", "Automotive Retail"), ("AZO", "AutoZone Inc", "Automotive Retail"),
        ("ROST", "Ross Stores", "Apparel Retail"), ("MAR", "Marriott International", "Hotels & Resorts"),
        ("HLT", "Hilton Worldwide", "Hotels & Resorts"), ("YUM", "Yum Brands", "Restaurants"),
        ("DHI", "DR Horton Inc", "Homebuilding"), ("LEN", "Lennar Corp", "Homebuilding"),
        ("GM", "General Motors", "Auto Manufacturers"), ("F", "Ford Motor", "Auto Manufacturers"),
        ("ABNB", "Airbnb Inc", "Internet Travel"), ("LULU", "Lululemon Athletica", "Apparel"),
        ("RCL", "Royal Caribbean", "Hotels & Resorts"), ("EBAY", "eBay Inc", "Internet Retail"),
        ("DG", "Dollar General", "General Merchandise"), ("DLTR", "Dollar Tree", "General Merchandise"),
        ("BBY", "Best Buy Co", "Computer & Electronics"), ("ULTA", "Ulta Beauty", "Specialty Stores"),
        ("POOL", "Pool Corp", "Distributors"), ("TSCO", "Tractor Supply", "Specialty Stores"),
    ],
    "Consumer Staples": [
        ("PG", "Procter & Gamble", "Household Products"), ("KO", "Coca-Cola Co", "Soft Drinks"),
        ("PEP", "PepsiCo Inc", "Soft Drinks"), ("COST", "Costco Wholesale", "Hypermarkets"),
        ("WMT", "Walmart Inc", "Hypermarkets"), ("PM", "Philip Morris International", "Tobacco"),
        ("MO", "Altria Group", "Tobacco"), ("MDLZ", "Mondelez International", "Packaged Foods"),
        ("CL", "Colgate-Palmolive", "Household Products"), ("KMB", "Kimberly-Clark", "Household Products"),
        ("GIS", "General Mills", "Packaged Foods"), ("K", "Kellanova", "Packaged Foods"),
        ("HSY", "Hershey Co", "Packaged Foods"), ("SJM", "JM Smucker", "Packaged Foods"),
        ("KHC", "Kraft Heinz", "Packaged Foods"), ("STZ", "Constellation Brands", "Distillers"),
        ("SYY", "Sysco Corp", "Food Distributors"), ("ADM", "Archer-Daniels-Midland", "Agricultural Products"),
        ("KR", "Kroger Co", "Food Retail"), ("WBA", "Walgreens Boots Alliance", "Drug Retail"),
        ("EL", "Estee Lauder", "Personal Products"), ("CHD", "Church & Dwight", "Household Products"),
        ("CLX", "Clorox Co", "Household Products"), ("MKC", "McCormick & Co", "Packaged Foods"),
        ("CAG", "Conagra Brands", "Packaged Foods"),
    ],
    "Energy": [
        ("XOM", "Exxon Mobil", "Integrated Oil & Gas"), ("CVX", "Chevron Corp", "Integrated Oil & Gas"),
        ("COP", "ConocoPhillips", "Oil & Gas Exploration"), ("EOG", "EOG Resources", "Oil & Gas Exploration"),
        ("SLB", "Schlumberger", "Oil & Gas Equipment"), ("PSX", "Phillips 66", "Oil & Gas Refining"),
        ("VLO", "Valero Energy", "Oil & Gas Refining"), ("MPC", "Marathon Petroleum", "Oil & Gas Refining"),
        ("OXY", "Occidental Petroleum", "Oil & Gas Exploration"), ("PXD", "Pioneer Natural Resources", "Oil & Gas Exploration"),
        ("HAL", "Halliburton", "Oil & Gas Equipment"), ("DVN", "Devon Energy", "Oil & Gas Exploration"),
        ("FANG", "Diamondback Energy", "Oil & Gas Exploration"), ("HES", "Hess Corp", "Oil & Gas Exploration"),
        ("BKR", "Baker Hughes", "Oil & Gas Equipment"), ("WMB", "Williams Companies", "Oil & Gas Storage"),
        ("KMI", "Kinder Morgan", "Oil & Gas Storage"), ("OKE", "ONEOK Inc", "Oil & Gas Storage"),
        ("TRGP", "Targa Resources", "Oil & Gas Storage"), ("EQT", "EQT Corp", "Oil & Gas Exploration"),
    ],
    "Industrials": [
        ("CAT", "Caterpillar Inc", "Construction Machinery"), ("BA", "Boeing Co", "Aerospace & Defense"),
        ("RTX", "RTX Corp", "Aerospace & Defense"), ("GE", "GE Aerospace", "Aerospace & Defense"),
        ("HON", "Honeywell International", "Industrial Conglomerates"), ("UPS", "United Parcel Service", "Air Freight"),
        ("DE", "Deere & Co", "Farm Machinery"), ("MMM", "3M Company", "Industrial Conglomerates"),
        ("LMT", "Lockheed Martin", "Aerospace & Defense"), ("NOC", "Northrop Grumman", "Aerospace & Defense"),
        ("GD", "General Dynamics", "Aerospace & Defense"), ("WM", "Waste Management", "Waste Management"),
        ("ETN", "Eaton Corp", "Electrical Components"), ("EMR", "Emerson Electric", "Electrical Components"),
        ("ITW", "Illinois Tool Works", "Industrial Machinery"), ("FDX", "FedEx Corp", "Air Freight"),
        ("CSX", "CSX Corp", "Railroads"), ("UNP", "Union Pacific", "Railroads"),
        ("NSC", "Norfolk Southern", "Railroads"), ("JCI", "Johnson Controls", "Building Products"),
        ("PH", "Parker-Hannifin", "Industrial Machinery"), ("ROK", "Rockwell Automation", "Electrical Components"),
        ("FAST", "Fastenal Co", "Industrial Distribution"), ("GWW", "WW Grainger", "Industrial Distribution"),
        ("CARR", "Carrier Global", "Building Products"), ("OTIS", "Otis Worldwide", "Building Products"),
        ("SWK", "Stanley Black & Decker", "Industrial Machinery"), ("IR", "Ingersoll Rand", "Industrial Machinery"),
        ("PCAR", "PACCAR Inc", "Construction Machinery"), ("CMI", "Cummins Inc", "Industrial Machinery"),
    ],
    "Utilities": [
        ("NEE", "NextEra Energy", "Electric Utilities"), ("DUK", "Duke Energy", "Electric Utilities"),
        ("SO", "Southern Company", "Electric Utilities"), ("D", "Dominion Energy", "Electric Utilities"),
        ("AEP", "American Electric Power", "Electric Utilities"), ("EXC", "Exelon Corp", "Electric Utilities"),
        ("SRE", "Sempra Energy", "Multi-Utilities"), ("XEL", "Xcel Energy", "Electric Utilities"),
        ("ED", "Consolidated Edison", "Electric Utilities"), ("WEC", "WEC Energy Group", "Electric Utilities"),
        ("ES", "Eversource Energy", "Electric Utilities"), ("DTE", "DTE Energy", "Electric Utilities"),
        ("FE", "FirstEnergy Corp", "Electric Utilities"), ("PPL", "PPL Corp", "Electric Utilities"),
        ("CMS", "CMS Energy", "Electric Utilities"), ("AWK", "American Water Works", "Water Utilities"),
        ("ATO", "Atmos Energy", "Gas Utilities"), ("NI", "NiSource Inc", "Multi-Utilities"),
        ("EVRG", "Evergy Inc", "Electric Utilities"), ("LNT", "Alliant Energy", "Electric Utilities"),
    ],
    "Materials": [
        ("LIN", "Linde PLC", "Industrial Gases"), ("APD", "Air Products", "Industrial Gases"),
        ("SHW", "Sherwin-Williams", "Specialty Chemicals"), ("FCX", "Freeport-McMoRan", "Copper"),
        ("NEM", "Newmont Corp", "Gold"), ("ECL", "Ecolab Inc", "Specialty Chemicals"),
        ("DD", "DuPont de Nemours", "Diversified Chemicals"), ("DOW", "Dow Inc", "Commodity Chemicals"),
        ("NUE", "Nucor Corp", "Steel"), ("PPG", "PPG Industries", "Specialty Chemicals"),
        ("CTVA", "Corteva Inc", "Fertilizers"), ("VMC", "Vulcan Materials", "Construction Materials"),
        ("MLM", "Martin Marietta", "Construction Materials"), ("ALB", "Albemarle Corp", "Specialty Chemicals"),
        ("CF", "CF Industries", "Fertilizers"), ("CE", "Celanese Corp", "Specialty Chemicals"),
        ("EMN", "Eastman Chemical", "Diversified Chemicals"), ("IFF", "International Flavors", "Specialty Chemicals"),
        ("PKG", "Packaging Corp", "Paper Packaging"), ("IP", "International Paper", "Paper Packaging"),
    ],
    "Real Estate": [
        ("AMT", "American Tower", "Telecom Tower REITs"), ("PLD", "Prologis Inc", "Industrial REITs"),
        ("SPG", "Simon Property Group", "Retail REITs"), ("CCI", "Crown Castle", "Telecom Tower REITs"),
        ("EQIX", "Equinix Inc", "Data Center REITs"), ("PSA", "Public Storage", "Self-Storage REITs"),
        ("O", "Realty Income", "Retail REITs"), ("DLR", "Digital Realty Trust", "Data Center REITs"),
        ("WELL", "Welltower Inc", "Health Care REITs"), ("AVB", "AvalonBay Communities", "Residential REITs"),
        ("EQR", "Equity Residential", "Residential REITs"), ("VICI", "VICI Properties", "Casino REITs"),
        ("ARE", "Alexandria Real Estate", "Office REITs"), ("MAA", "Mid-America Apartment", "Residential REITs"),
        ("UDR", "UDR Inc", "Residential REITs"), ("ESS", "Essex Property", "Residential REITs"),
        ("HST", "Host Hotels", "Hotel REITs"), ("CPT", "Camden Property", "Residential REITs"),
        ("EXR", "Extra Space Storage", "Self-Storage REITs"), ("REG", "Regency Centers", "Retail REITs"),
    ],
    "Communication Services": [
        ("GOOGL", "Alphabet Inc", "Interactive Media"), ("META", "Meta Platforms", "Interactive Media"),
        ("NFLX", "Netflix Inc", "Movies & Entertainment"), ("DIS", "Walt Disney", "Movies & Entertainment"),
        ("CMCSA", "Comcast Corp", "Cable & Satellite"), ("VZ", "Verizon Communications", "Integrated Telecom"),
        ("T", "AT&T Inc", "Integrated Telecom"), ("TMUS", "T-Mobile US", "Wireless Telecom"),
        ("CHTR", "Charter Communications", "Cable & Satellite"), ("WBD", "Warner Bros Discovery", "Movies & Entertainment"),
        ("EA", "Electronic Arts", "Interactive Entertainment"), ("TTWO", "Take-Two Interactive", "Interactive Entertainment"),
        ("RBLX", "Roblox Corp", "Interactive Entertainment"), ("SPOT", "Spotify Technology", "Interactive Media"),
        ("PINS", "Pinterest Inc", "Interactive Media"), ("SNAP", "Snap Inc", "Interactive Media"),
        ("MTCH", "Match Group", "Interactive Media"), ("ZM", "Zoom Video", "Application Software"),
        ("ROKU", "Roku Inc", "Movies & Entertainment"), ("LYV", "Live Nation", "Movies & Entertainment"),
    ],
}


def generate_synthetic_dataset(seed=42):
    """
    Generate a realistic synthetic dataset based on known industry financial profiles.
    Each company gets ratios drawn from its sector's distribution with some noise.
    """
    np.random.seed(seed)
    records = []

    for sector, companies in SECTOR_COMPANIES.items():
        profile = SECTOR_PROFILES[sector]

        for ticker, name, sub_industry in companies:
            record = {
                "ticker": ticker,
                "company_name": name,
                "sector": sector,
                "industry": sub_industry,
            }

            for ratio_name in RATIO_NAMES:
                mean, std = profile[ratio_name]
                # Add company-specific variation
                value = np.random.normal(mean, std)
                # Clip to reasonable range
                if "Margin" in ratio_name or "to_" in ratio_name:
                    value = np.clip(value, -0.5, 1.5)
                record[ratio_name] = round(value, 4)

            records.append(record)

    return pd.DataFrame(records)


def try_live_build():
    """Attempt to build dataset from yfinance (requires internet)."""
    try:
        import yfinance as yf
        # Quick test
        t = yf.Ticker("AAPL")
        _ = t.info
        return True
    except Exception:
        return False


def build_dataset(force_synthetic=False):
    print("=" * 60)
    print("Financial DNA — Building Training Dataset")
    print("=" * 60)

    if not force_synthetic and try_live_build():
        print("\nyfinance is available — pulling live data...")
        print("(This will take several minutes for 500+ companies)")
        # Import and run the live fetcher
        from utils.data_fetcher import get_sp500_tickers, fetch_company_data
        sp500 = get_sp500_tickers()
        records = []
        for i, row in sp500.iterrows():
            ticker = row["Symbol"]
            if (i + 1) % 25 == 0:
                print(f"  Processing {i + 1}/{len(sp500)}: {ticker}...")
            data = fetch_company_data(ticker)
            if data:
                records.append(data)
            if (i + 1) % 50 == 0:
                time.sleep(1)
        df = pd.DataFrame(records)
    else:
        print("\nUsing synthetic data based on real industry financial profiles.")
        print("(Run on your local machine with internet to use live yfinance data)")
        df = generate_synthetic_dataset()

    # Clean up
    df = df[df["sector"] != "Unknown"]

    # Impute and clip
    for col in RATIO_NAMES:
        sector_medians = df.groupby("sector")[col].transform("median")
        df[col] = df[col].fillna(sector_medians)
        df[col] = df[col].fillna(df[col].median())
        df[col] = df[col].fillna(0.0)
        mean, std = df[col].mean(), df[col].std()
        if std > 0:
            df[col] = df[col].clip(mean - 3 * std, mean + 3 * std)

    os.makedirs(DATA_DIR, exist_ok=True)
    output_path = os.path.join(DATA_DIR, "company_ratios.csv")
    df.to_csv(output_path, index=False)

    print(f"\nDataset saved to: {output_path}")
    print(f"Shape: {df.shape}")
    print(f"\nSector distribution:")
    print(df["sector"].value_counts().to_string())
    print("\n" + "=" * 60)
    print("Dataset build complete!")
    print("=" * 60)

    return df


if __name__ == "__main__":
    force = "--synthetic" in sys.argv
    build_dataset(force_synthetic=force)
