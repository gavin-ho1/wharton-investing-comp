# src/portfolio_beta.py
# This script will calculate the overall portfolio beta.

import pandas as pd
import yfinance as yf
import yaml
from datetime import datetime, timedelta
from src.financial_metrics import calculate_beta_from_returns

def get_config():
    """Reads the config.yaml file."""
    with open('config.yaml', 'r') as file:
        return yaml.safe_load(file)

def calculate_portfolio_beta():
    """
    Downloads necessary data and uses the shared financial_metrics module
    to calculate the overall portfolio beta based on the optimized allocations.
    """
    print("--- Calculating Portfolio Beta ---")
    
    # 1. Load configuration and portfolio weights
    config = get_config()
    benchmark_ticker = config['quant_factor_analysis']['benchmark_ticker']
    lookback_years = config['correlation_analysis']['lookback_years']
    weights_filepath = f"data/{config['portfolio_optimization']['output_filename']}"
    
    try:
        allocations_df = pd.read_csv(weights_filepath, index_col=0)
    except FileNotFoundError:
        print(f"Error: Could not find the allocations file at {weights_filepath}")
        return

    # 2. Prepare list of tickers and date range for data download
    portfolio_tickers = allocations_df.index.tolist()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_years * 365)
    
    # 3. Download benchmark and portfolio data
    print(f"Downloading benchmark data: {benchmark_ticker}...")
    try:
        benchmark_prices = yf.download(benchmark_ticker, start=start_date, end=end_date, progress=False)['Close']
    except Exception as e:
        print(f"An exception occurred while downloading benchmark data: {e}")
        return

    print(f"Downloading price data for {len(portfolio_tickers)} portfolio tickers...")
    portfolio_prices = yf.download(portfolio_tickers, start=start_date, end=end_date, progress=False)
    
    if portfolio_prices.empty:
        print("Error: Could not download any price data for portfolio tickers.")
        return
        
    # Use 'Close' prices for calculations
    if 'Close' in portfolio_prices.columns:
        portfolio_prices = portfolio_prices['Close']

    # 4. Calculate daily returns
    benchmark_returns = benchmark_prices.pct_change().dropna()
    portfolio_returns_df = portfolio_prices.pct_change().dropna()

    # 5. Calculate portfolio beta using the shared function
    portfolio_beta = calculate_beta_from_returns(portfolio_returns_df, benchmark_returns, allocations_df)
    
    if portfolio_beta is None:
        print("\nPortfolio Beta calculation failed.")
        return None

    print("\n--- Portfolio Beta Calculation Complete ---")
    print(f"Lookback Period: {lookback_years} years")
    print(f"Benchmark: {benchmark_ticker}")
    print(f"Overall Portfolio Beta: {portfolio_beta:.4f}")
    
    return portfolio_beta

if __name__ == "__main__":
    calculate_portfolio_beta()
