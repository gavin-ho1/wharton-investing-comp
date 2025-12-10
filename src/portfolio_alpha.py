# src/portfolio_alpha.py
# This script will calculate the overall portfolio alpha.

import pandas as pd
import yfinance as yf
import yaml
from datetime import datetime, timedelta
from src.financial_metrics import calculate_beta_from_returns

def get_config():
    """Reads the config.yaml file."""
    with open('config.yaml', 'r') as file:
        return yaml.safe_load(file)

def calculate_portfolio_alpha():
    """
    Downloads portfolio, benchmark, and risk-free rate data, then uses the
    shared financial_metrics module to calculate beta and finally computes alpha.
    Alpha = R_p - (R_f + Beta * (R_m - R_f))
    """
    print("--- Calculating Portfolio Alpha ---")
    
    # 1. Load configuration and portfolio weights
    config = get_config()
    benchmark_ticker = config['quant_factor_analysis']['benchmark_ticker']
    risk_free_ticker = config['quant_factor_analysis']['risk_free_ticker']
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
    
    # 3. Download all necessary data in one go
    print("Downloading financial data...")
    all_tickers = [benchmark_ticker, risk_free_ticker] + portfolio_tickers
    all_data = yf.download(all_tickers, start=start_date, end=end_date, progress=False)
    
    if all_data.empty:
        print("Error: Could not download any financial data.")
        return

    prices = all_data['Close']
    benchmark_prices = prices[benchmark_ticker]
    risk_free_data = prices[risk_free_ticker]
    portfolio_prices = prices[portfolio_tickers]

    # 4. Calculate daily returns
    benchmark_returns = benchmark_prices.pct_change().dropna()
    portfolio_returns_df = portfolio_prices.pct_change().dropna()

    # 5. Calculate Portfolio Beta using the shared, reusable function
    print("Calculating portfolio beta...")
    portfolio_beta = calculate_beta_from_returns(portfolio_returns_df, benchmark_returns, allocations_df)
    
    if portfolio_beta is None:
        print("\nCould not calculate portfolio alpha because beta calculation failed.")
        return None

    # 6. Calculate Annualized Returns and Risk-Free Rate
    # The T-bill rate from YFinance ('^IRX') is a yield percentage.
    # We take the mean of the series and divide by 100 to get the decimal rate.
    risk_free_rate_annualized = risk_free_data.mean() / 100
    if pd.isna(risk_free_rate_annualized):
        print("Warning: Could not calculate average risk-free rate. Defaulting to 0.")
        risk_free_rate_annualized = 0.0

    weights_series = allocations_df['Weight']
    valid_tickers = [t for t in weights_series.index if t in portfolio_returns_df.columns]
    aligned_weights = weights_series[valid_tickers]
    
    portfolio_daily_returns = (portfolio_returns_df[valid_tickers] * aligned_weights).sum(axis=1)

    # Align the final portfolio return series with the benchmark to ensure calculations
    # are based on the same dates before annualizing.
    aligned_portfolio_returns, aligned_benchmark_returns = portfolio_daily_returns.align(benchmark_returns, join='inner')

    portfolio_return_annualized = (1 + aligned_portfolio_returns.mean())**252 - 1
    benchmark_return_annualized = (1 + aligned_benchmark_returns.mean())**252 - 1

    # 7. Calculate portfolio alpha
    expected_return = risk_free_rate_annualized + portfolio_beta * (benchmark_return_annualized - risk_free_rate_annualized)
    portfolio_alpha = portfolio_return_annualized - expected_return

    print("\n--- Portfolio Alpha Calculation Complete ---")
    print(f"Lookback Period: {lookback_years} years")
    print(f"Benchmark: {benchmark_ticker}")
    print(f"Risk-Free Rate Proxy: {risk_free_ticker}")
    print("-" * 20)
    print(f"Annualized Portfolio Return: {portfolio_return_annualized:.4%}")
    print(f"Annualized Benchmark Return: {benchmark_return_annualized:.4%}")
    print(f"Annualized Avg. Risk-Free Rate: {risk_free_rate_annualized:.4%}")
    print(f"Portfolio Beta: {portfolio_beta:.4f}")
    print(f"Expected Return (CAPM): {expected_return:.4%}")
    print(f"Overall Portfolio Alpha: {portfolio_alpha:.4%}")
    
    return portfolio_alpha

if __name__ == "__main__":
    calculate_portfolio_alpha()
