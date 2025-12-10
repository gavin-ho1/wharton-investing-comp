# src/data_collection.py
# This module contains functions for Phase 1: Data Collection & Preparation.

import yfinance as yf
import pandas as pd
from currency_converter import CurrencyConverter
from datetime import datetime, timedelta
import os
import yaml

def load_tickers(filepath="tickers.txt"):
    """Loads tickers from a text file, one per line."""
    print(f"Loading tickers from {filepath}...")
    with open(filepath, 'r') as f:
        tickers = [line.strip() for line in f.readlines() if line.strip()]
    print(f"Loaded {len(tickers)} tickers.")
    return tickers

def fetch_price_and_dividend_data(tickers, lookback_years=10):
    """Fetches historical adjusted close prices and dividends for a list of tickers."""
    print(f"Fetching historical price and dividend data for {len(tickers)} tickers...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_years * 365)
    
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)
    prices = data['Adj Close']
    
    if 'Dividends' in data.columns:
        dividends = data['Dividends']
    else:
        print("  - Warning: No dividend data found. Creating an empty dividend dataframe.")
        dividends = pd.DataFrame(index=prices.index, columns=prices.columns).fillna(0)

    print(f"Successfully fetched data from {start_date.date()} to {end_date.date()}.")
    return prices, dividends

def fetch_fundamental_data(tickers, etf_list):
    """Fetches fundamental data for a list of tickers, skipping ETFs."""
    print(f"Fetching fundamental data for {len(tickers)} tickers...")
    fundamentals = {}
    tickers_to_fetch = [t for t in tickers if t not in etf_list]
    for ticker in tickers_to_fetch:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            fundamentals[ticker] = {
                'info': info,
                'financials': stock.financials,
                'balance_sheet': stock.balance_sheet,
                'cashflow': stock.cashflow
            }
            print(f"  Successfully fetched fundamentals for {ticker}.")
        except Exception as e:
            print(f"  Could not fetch fundamental data for {ticker}: {e}")
    return fundamentals

def convert_prices_to_usd(prices_df, fundamentals_data, etf_list):
    """Converts a DataFrame of prices to a target currency (USD)."""
    print("Converting prices to USD...")
    cc = CurrencyConverter()
    prices_usd = prices_df.copy()
    
    for ticker in prices_df.columns:
        if ticker in etf_list: continue
        try:
            currency = fundamentals_data[ticker]['info'].get('currency', 'USD').upper()
            if currency != 'USD':
                print(f"  Converting {ticker} from {currency} to USD...")
                rate = cc.convert(1, currency, 'USD')
                prices_usd[ticker] *= rate
        except Exception as e:
            print(f"  Could not convert currency for {ticker}: {e}. Assuming USD.")
            
    return prices_usd

def clean_data(df):
    """Cleans a dataframe by forward-filling and dropping empty columns."""
    print("Cleaning data...")
    cleaned_df = df.ffill()
    cleaned_df = cleaned_df.dropna(axis=1, how='all')
    print("Data cleaning complete.")
    return cleaned_df

def save_data(prices_df, dividends_df, fundamentals_data, config):
    """Saves all processed data to the specified output directory."""
    output_dir = config['output_dir']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    prices_path = os.path.join(output_dir, config['prices_filename'])
    prices_df.to_csv(prices_path)
    print(f"Price data saved to {prices_path}")
    
    dividends_path = os.path.join(output_dir, config['dividends_filename'])
    dividends_df.to_csv(dividends_path)
    print(f"Dividend data saved to {dividends_path}")
    
    timestamp = datetime.now().strftime('%Y-%m-%d')
    fundamentals_filename = f"{config['fundamentals_filename_prefix']}_{timestamp}.pkl"
    fundamentals_path = os.path.join(output_dir, fundamentals_filename)
    pd.to_pickle(fundamentals_data, fundamentals_path)
    print(f"Fundamental data saved to {fundamentals_path}")

def load_price_data(data_dir, prices_filename):
    """Loads the price data CSV."""
    prices_path = os.path.join(data_dir, prices_filename)
    return pd.read_csv(prices_path, index_col='Date', parse_dates=True)

# src/data_collection.py
# Add this function and modify run_data_collection

def load_always_include_tickers(config_path="config.yaml"):
    """Loads always-include tickers from the config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        screening_config = config['fundamental_screening']
    
    always_include_file = screening_config['always_include_filename']
    with open(always_include_file, 'r') as f:
        return [line.strip() for line in f.readlines() if line.strip() and not line.startswith('#')]

def run_data_collection(config_path="config.yaml"):
    """Main function to run the entire data collection and preparation phase."""
    print("--- Starting Phase 1: Data Collection & Preparation ---")
    
    with open(config_path, 'r') as f:
        config_full = yaml.safe_load(f)
        config = config_full['data_collection']
        benchmark_ticker = config_full['quant_factor_analysis']['benchmark_ticker']
        
    tickers = load_tickers()
    always_include_tickers = load_always_include_tickers(config_path)
    
    with open(config['etf_list_filename'], 'r') as f:
        etf_list = [line.strip() for line in f.readlines() if line.strip()]

    # Include always-include stocks in data collection
    full_ticker_list = list(set(tickers + [benchmark_ticker] + always_include_tickers))

    prices, dividends = fetch_price_and_dividend_data(full_ticker_list, config['lookback_years'])
    fundamentals = fetch_fundamental_data(full_ticker_list, etf_list)
    
    prices_usd = convert_prices_to_usd(prices, fundamentals, etf_list)
    
    cleaned_prices = clean_data(prices_usd)
    cleaned_dividends = clean_data(dividends)
    
    save_data(cleaned_prices, cleaned_dividends, fundamentals, config)
    
    print("--- Phase 1 Finished ---")