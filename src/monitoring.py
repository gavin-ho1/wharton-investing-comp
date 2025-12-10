# src/monitoring.py
# This module contains the logic for generating a recent performance report.

import pandas as pd
import yaml
import os
import quantstats as qs
import yfinance as yf
from datetime import datetime, timedelta
from .data_collection import load_price_data
from .portfolio_optimization import load_optimized_weights

def run_monitoring(config_path="config.yaml"):
    """
    Main function to run the monitoring report generation.
    """
    print("\n--- Starting Phase 8: Monitoring & Reporting ---")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        data_config = config['data_collection']
        opt_config = config['portfolio_optimization']
        monitor_config = config['monitoring']
        backtest_config = config['backtesting'] # For benchmark ticker

    # Load data
    weights_df = load_optimized_weights(data_config['output_dir'], opt_config['output_filename'])
    prices_df = load_price_data(data_config['output_dir'], data_config['prices_filename'])
    
    # Filter for monitoring period
    end_date = prices_df.index.max()
    start_date = end_date - timedelta(days=monitor_config['lookback_period_days'])
    
    prices_period = prices_df.loc[start_date:end_date]
    weights = weights_df['Weight']
    
    # Calculate portfolio returns
    daily_returns = prices_period[weights.index].pct_change().dropna()
    portfolio_daily_returns = (daily_returns * weights).sum(axis=1)
    portfolio_daily_returns = portfolio_daily_returns.asfreq('D', fill_value=0)
    
    # Generate report
    print("Generating monitoring report...")
    benchmark_ticker = backtest_config['benchmark_ticker']
    benchmark = yf.download(benchmark_ticker, start=start_date, end=end_date, auto_adjust=False)['Adj Close'].pct_change()
    
    report_path = os.path.join(data_config['output_dir'], monitor_config['report_filename'])
    
    try:
        qs.reports.html(portfolio_daily_returns, benchmark=benchmark, output=report_path, title='Portfolio Monitoring Report (Last Year)', period='daily')
        print(f"Monitoring report saved to {report_path}")
    except ValueError as e:
        print(f"  - Warning: Could not generate full HTML report due to an error: {e}")
        print("  - Generating a basic report instead.")
        basic_report_path = os.path.join(data_config['output_dir'], 'basic_' + monitor_config['report_filename'])
        qs.reports.basic(portfolio_daily_returns, benchmark=benchmark, output=basic_report_path, title='Portfolio Monitoring Report (Basic)')
        print(f"Basic monitoring report saved to {basic_report_path}")

    print("\n--- Phase 8 Finished ---")
