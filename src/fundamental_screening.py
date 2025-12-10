# src/fundamental_screening.py
# This module contains functions for Phase 2: Fundamental Screening.

import pandas as pd
import numpy as np
import yaml
import os
from glob import glob
from currency_converter import CurrencyConverter

# --- Data Loading ---

def load_latest_fundamental_data(data_dir):
    """Loads the most recent fundamentals .pkl file."""
    print("Loading latest fundamental data...")
    fundamental_files = glob(os.path.join(data_dir, 'fundamentals_*.pkl'))
    if not fundamental_files:
        raise FileNotFoundError("No fundamental data files found.")
    latest_file = max(fundamental_files, key=os.path.getctime)
    return pd.read_pickle(latest_file)

def load_price_data(data_dir, prices_filename):
    """Loads the price data CSV."""
    prices_path = os.path.join(data_dir, prices_filename)
    return pd.read_csv(prices_path, index_col='Date', parse_dates=True)

# --- Metric Calculation Helpers (Re-implemented) ---

def calculate_pe_ratio(info, financials):
    try:
        market_cap = info.get('marketCap')
        net_income = financials.loc['Net Income'].iloc[0]
        if market_cap and net_income > 0: return market_cap / net_income
    except (KeyError, IndexError): pass
    return np.nan

def calculate_pb_ratio(info, balance_sheet):
    try:
        market_cap = info.get('marketCap')
        book_value = balance_sheet.loc['Total Stockholder Equity'].iloc[0]
        if market_cap and book_value > 0: return market_cap / book_value
    except (KeyError, IndexError): pass
    return np.nan

def calculate_ev_ebitda(info, financials, balance_sheet, cashflow):
    try:
        market_cap = info.get('marketCap')
        total_debt = balance_sheet.loc['Total Liab'].iloc[0]
        cash = balance_sheet.loc['Cash'].iloc[0]
        enterprise_value = market_cap + total_debt - cash
        ebit = financials.loc['Ebit'].iloc[0]
        depreciation = cashflow.loc['Depreciation'].iloc[0]
        ebitda = ebit + depreciation
        if enterprise_value and ebitda > 0: return enterprise_value / ebitda
    except (KeyError, IndexError): pass
    return np.nan

def calculate_cagr(series, years=5):
    series = series.dropna().iloc[::-1]
    if len(series) < 2: return np.nan
    num_years_available = min(years, len(series) - 1)
    start_value, end_value = series.iloc[0], series.iloc[num_years_available]
    if start_value <= 0 or end_value <= 0: return np.nan
    return (end_value / start_value) ** (1 / num_years_available) - 1

def calculate_revenue_cagr(financials):
    try: return calculate_cagr(financials.loc['Total Revenue'])
    except (KeyError, IndexError): return np.nan

def calculate_eps_cagr(financials):
    try: return calculate_cagr(financials.loc['Basic EPS'])
    except (KeyError, IndexError): return np.nan

def calculate_roe(financials, balance_sheet):
    try:
        net_income = financials.loc['Net Income'].iloc[0]
        equity = balance_sheet.loc['Total Stockholder Equity'].iloc[0]
        if equity > 0: return net_income / equity
    except (KeyError, IndexError): pass
    return np.nan
    
def calculate_all_metrics(tickers, fundamentals_data):
    metrics = []
    for ticker in tickers:
        if ticker not in fundamentals_data: continue
        data = fundamentals_data[ticker]
        info, financials, balance_sheet, cashflow = data.get('info', {}), data.get('financials'), data.get('balance_sheet'), data.get('cashflow')
        if any(df is None or df.empty for df in [financials, balance_sheet, cashflow]): continue
        metrics.append({
            'Ticker': ticker,
            'P/E': calculate_pe_ratio(info, financials),
            'P/B': calculate_pb_ratio(info, balance_sheet),
            'EV/EBITDA': calculate_ev_ebitda(info, financials, balance_sheet, cashflow),
            'Revenue CAGR (5Y)': calculate_revenue_cagr(financials),
            'EPS CAGR (5Y)': calculate_eps_cagr(financials),
            'ROE': calculate_roe(financials, balance_sheet),
        })
    return pd.DataFrame(metrics).set_index('Ticker')

def rank_metrics(metrics_df):
    ranked_df = pd.DataFrame(index=metrics_df.index)
    lower_is_better = ['P/E', 'P/B', 'EV/EBITDA']
    for metric in metrics_df.columns:
        rank_col = metrics_df[metric].rank(pct=True)
        if metric in lower_is_better: rank_col = 1 - rank_col
        ranked_df[f"{metric}_Rank"] = rank_col.fillna(0.5)
    return ranked_df

def calculate_composite_score(ranked_df, weights):
    metric_categories = {
        'Value': ['P/E_Rank', 'P/B_Rank', 'EV/EBITDA_Rank'],
        'Growth': ['Revenue CAGR (5Y)_Rank', 'EPS CAGR (5Y)_Rank'],
        'Quality': ['ROE_Rank'],
        'Stability': []
    }
    scored_df = ranked_df.copy()
    scored_df['Fundamental_Score'] = 0
    for category, category_metrics in metric_categories.items():
        available_metrics = [m for m in category_metrics if m in scored_df.columns]
        if not available_metrics: continue
        category_score = scored_df[available_metrics].mean(axis=1)
        scored_df['Fundamental_Score'] += weights[category] * category_score
    return scored_df

def filter_by_price(prices_df, fundamentals_data, min_price_usd, etf_list):
    """
    Filters out stocks with a price below the minimum USD threshold,
    handling currency conversion. ETFs without fundamental data are assumed to be in USD.
    """
    print(f"Filtering out stocks below ${min_price_usd} USD...")
    c = CurrencyConverter(fallback_on_missing_rate=True)
    last_prices = prices_df.iloc[-1]
    to_keep = []
    to_discard = []

    for ticker, price in last_prices.items():
        currency = 'USD' # Default currency
        has_fundamentals = ticker in fundamentals_data and 'info' in fundamentals_data[ticker]

        if has_fundamentals:
            currency = fundamentals_data[ticker]['info'].get('currency', 'USD').upper()
        elif ticker in etf_list:
            # This is an ETF without fundamental data, assume its price is in USD
            pass
        else:
            # This is a regular stock without fundamental data, discard it
            print(f"  - Warning: No fundamental info for non-ETF {ticker}, discarding.")
            to_discard.append(ticker)
            continue

        try:
            price_usd = c.convert(price, currency, 'USD') if currency != 'USD' else price
            if price_usd >= min_price_usd:
                to_keep.append(ticker)
            else:
                to_discard.append(f"{ticker} (${price_usd:.2f})")
        except Exception as e:
            print(f"  - Warning: Could not convert price for {ticker} from {currency}. Error: {e}. Discarding.")
            to_discard.append(ticker)

    if to_discard:
        print(f"  - Discarding {len(to_discard)} stocks: {', '.join(to_discard)}")
        
    return to_keep

# src/fundamental_screening.py
# Modify the ETF handling section:

def run_fundamental_screening(config_path="config.yaml"):
    """Main function to run the entire fundamental screening phase."""
    print("\n--- Starting Phase 2: Fundamental Screening ---")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        data_config = config['data_collection']
        screening_config = config['fundamental_screening']

    fundamentals = load_latest_fundamental_data(data_config['output_dir'])
    prices = load_price_data(data_config['output_dir'], data_config['prices_filename'])
    with open(data_config['etf_list_filename'], 'r') as f:
        etf_list = [line.strip() for line in f.readlines() if line.strip()]

    # --- 1. Price Filtering ---
    price_filtered_tickers = filter_by_price(prices, fundamentals, data_config['min_stock_price'], etf_list)
    
    # --- 2. Separate different ticker types ---
    etfs_after_price_filter = [t for t in price_filtered_tickers if t in etf_list]
    stocks_to_screen = [t for t in price_filtered_tickers if t not in etf_list]
    
    print(f"After price filtering: {len(stocks_to_screen)} stocks, {len(etfs_after_price_filter)} ETFs")
    
    # --- 3. Fundamental Screening for regular stocks ---
    metrics_df = calculate_all_metrics(stocks_to_screen, fundamentals)
    ranked_df = rank_metrics(metrics_df)
    scored_df = calculate_composite_score(ranked_df, screening_config['score_weights'])
    
    # Create a DataFrame for all tickers with scores
    full_scores_df = metrics_df.join(scored_df).copy()

    # --- 4. Handle ETFs: Assign neutral score ---
    # All ETFs that pass price filter get included with neutral score
    etf_scores_df = pd.DataFrame(index=etfs_after_price_filter)
    etf_scores_df['Fundamental_Score'] = 0.5
    
    # Combine scores of regular stocks and ETFs
    all_scored_df = pd.concat([scored_df, etf_scores_df])

    # --- 5. Apply screening percentile cutoff to ALL tickers (stocks + ETFs) ---
    cutoff = all_scored_df['Fundamental_Score'].quantile(screening_config['screening_percentile_cutoff'])
    screened_tickers = all_scored_df[all_scored_df['Fundamental_Score'] >= cutoff].index.tolist()
    
    # Update the full scores df for saving
    full_scores_df = pd.concat([metrics_df.join(scored_df), etf_scores_df])
    
    print(f"Screening complete. {len(screened_tickers)} tickers passed.")
    print(f"  - Stocks: {len([t for t in screened_tickers if t not in etf_list])}")
    print(f"  - ETFs: {len([t for t in screened_tickers if t in etf_list])}")
    print(f"  - Note: Always-include tickers will be added in Phase 4 after all filtering")

    # --- 6. Save outputs ---
    full_scores_df.to_csv(os.path.join(data_config['output_dir'], screening_config['output_scores_filename']))
    with open(os.path.join(data_config['output_dir'], screening_config['output_tickers_filename']), 'w') as f:
        for ticker in screened_tickers: 
            f.write(f"{ticker}\n")

    print("--- Phase 2 Finished ---")