# src/backtesting.py
# This module contains the logic for running the historical backtest.

import pandas as pd
import numpy as np
import yaml
import os
import quantstats as qs
import yfinance as yf
from .strategy import get_target_weights_for_date
from .data_collection import load_price_data
from .fundamental_screening import load_latest_fundamental_data

def run_backtest(config_path="config.yaml"):
    """
    Main function to run the entire backtesting and performance evaluation phase.
    """
    print("\n--- Starting Phase 7: Backtesting & Performance Evaluation ---")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        data_config = config['data_collection']
        backtest_config = config['backtesting']

    # Load all historical data
    all_prices = load_price_data(data_config['output_dir'], data_config['prices_filename'])
    all_dividends = pd.read_csv(os.path.join(data_config['output_dir'], data_config['dividends_filename']), index_col='Date', parse_dates=True)
    all_fundamentals = load_latest_fundamental_data(data_config['output_dir'])
    
    start_date = pd.to_datetime(backtest_config['start_date'])
    end_date = all_prices.index.max()
    
    portfolio_value = backtest_config.get('initial_investment', 500000)
    cash = portfolio_value
    holdings = pd.Series(dtype='float64')
    
    portfolio_history = pd.Series(dtype='float64')
    dividend_log = []

    rebalance_dates = pd.date_range(start=start_date, end=end_date, freq='YS')

    for i, rebalance_date in enumerate(rebalance_dates):
        print(f"\nRebalancing on {rebalance_date.date()}...")
        
        if not holdings.empty:
            last_price_idx = all_prices.index.searchsorted(rebalance_date - pd.Timedelta(days=1), side='right') - 1
            last_prices = all_prices.iloc[last_price_idx][holdings.index]
            portfolio_value = (holdings * last_prices).sum() + cash
        
        target_weights = get_target_weights_for_date(rebalance_date, all_prices, all_fundamentals, config)
        
        rebalance_price_idx = all_prices.index.searchsorted(rebalance_date, side='left')
        rebalance_price_date = all_prices.index[rebalance_price_idx]
        current_prices = all_prices.iloc[rebalance_price_idx]

        target_positions_value = pd.Series(target_weights) * portfolio_value
        new_holdings = target_positions_value / current_prices.reindex(target_positions_value.index)
        
        trades = new_holdings.subtract(holdings, fill_value=0)
        transaction_costs = (trades.abs() * current_prices.reindex(trades.index)).sum() * (backtest_config['transaction_costs_bps'] / 10000)
        
        holdings = new_holdings.dropna()
        cash = portfolio_value - (holdings * current_prices.reindex(holdings.index)).sum() - transaction_costs
        
        start_period = rebalance_price_date
        end_period = rebalance_dates[i+1] if i + 1 < len(rebalance_dates) else end_date
        
        for date in pd.date_range(start_period, end_period):
            if date not in all_prices.index or date not in all_dividends.index: continue
            
            market_value = (holdings * all_prices.loc[date, holdings.index]).sum()
            portfolio_history[date] = market_value + cash
            
            daily_dividends = all_dividends.loc[date, holdings.index]
            if daily_dividends.sum() > 0:
                cash_from_dividends = (holdings * daily_dividends).sum()
                cash += cash_from_dividends
                dividend_log.append({'Date': date, 'Amount': cash_from_dividends})
                
            if date.year >= start_date.year + 3 and date.is_year_end:
                cash -= backtest_config['annual_withdrawal']

    portfolio_returns = portfolio_history.pct_change().dropna()
    portfolio_returns = portfolio_returns.asfreq('D', fill_value=0)
    benchmark = yf.download(backtest_config['benchmark_ticker'], start=start_date, end=end_date, auto_adjust=False)['Adj Close'].pct_change()
    
    report_path = os.path.join(data_config['output_dir'], backtest_config['report_filename'])
    qs.reports.html(portfolio_returns, benchmark=benchmark, output=report_path, title='Strategy Backtest Report', period='daily')
    print(f"\nBacktest report saved to {report_path}")

    # Dividend Coverage Report
    if dividend_log:
        dividend_df = pd.DataFrame(dividend_log).set_index('Date')
        annual_dividends = dividend_df.resample('A').sum()
    else:
        annual_dividends = pd.DataFrame(columns=['Amount'])

    coverage_report = []
    for year in range(start_date.year + 3, end_date.year + 1):
        try:
            dividends_this_year = annual_dividends.loc[f"{year}-12-31", 'Amount'] if not annual_dividends.empty else 0
        except KeyError:
            dividends_this_year = 0
        withdrawal = backtest_config['annual_withdrawal']
        coverage = (dividends_this_year / withdrawal) * 100 if withdrawal > 0 else 0
        coverage_report.append({'Year': year, 'Dividends': dividends_this_year, 'Withdrawal': withdrawal, 'Coverage (%)': coverage})

    coverage_df = pd.DataFrame(coverage_report)
    coverage_path = os.path.join(data_config['output_dir'], backtest_config['dividend_report_filename'])
    coverage_df.to_csv(coverage_path, index=False)
    print(f"Dividend coverage report saved to {coverage_path}")
    
    final_allocations_df = pd.DataFrame.from_dict(target_weights, orient='index', columns=['Weight'])
    allocations_path = os.path.join(data_config['output_dir'], backtest_config['final_allocations_filename'])
    final_allocations_df.to_csv(allocations_path)
    print(f"Final allocations saved to {allocations_path}")

    print("\n--- Phase 7 Finished ---")
    return portfolio_returns
