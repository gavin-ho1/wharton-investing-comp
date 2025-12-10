# src/financial_metrics.py
# This module contains shared financial calculation functions.

import pandas as pd
import numpy as np

def calculate_beta_from_returns(portfolio_returns_df, benchmark_returns, allocations_df):
    """
    Calculates the overall portfolio beta using pre-fetched returns data.

    Args:
        portfolio_returns_df (pd.DataFrame): DataFrame of daily returns for portfolio assets.
        benchmark_returns (pd.Series): Series of daily returns for the benchmark.
        allocations_df (pd.DataFrame): DataFrame with 'Weight' column for each asset.

    Returns:
        float: The calculated weighted portfolio beta. Returns None if calculation fails.
    """
    # 1. Calculate beta for each individual stock
    betas = {}
    portfolio_tickers = allocations_df.index.tolist()
    valid_tickers = [t for t in portfolio_tickers if t in portfolio_returns_df.columns and not portfolio_returns_df[t].isnull().all()]

    for ticker in valid_tickers:
        stock_returns = portfolio_returns_df[ticker]
        
        # Align returns to ensure we are comparing the same time periods
        aligned_stock, aligned_benchmark = stock_returns.align(benchmark_returns, join='inner')
        
        # Need at least 2 data points to calculate covariance
        if len(aligned_stock) < 2:
            betas[ticker] = np.nan
            continue
            
        # Calculate covariance and benchmark variance using population formula (ddof=0) for consistency
        covariance = np.cov(aligned_stock.to_numpy().flatten(), aligned_benchmark.to_numpy().flatten(), ddof=0)[0, 1]
        benchmark_variance = np.var(aligned_benchmark.to_numpy().flatten(), ddof=0)

        # Calculate beta
        if benchmark_variance > 0:
            beta = covariance / benchmark_variance
            betas[ticker] = beta
        else:
            betas[ticker] = np.nan

    if not betas:
        print("Warning: Could not calculate beta for any tickers.")
        return None

    betas_df = pd.DataFrame.from_dict(betas, orient='index', columns=['Beta'])
    
    # 2. Merge betas with allocations
    merged_df = allocations_df.join(betas_df)
    merged_df.dropna(subset=['Weight', 'Beta'], inplace=True)

    if merged_df.empty:
        print("Warning: No valid tickers with both weights and beta found.")
        return None

    # 3. Calculate the weighted portfolio beta
    merged_df['Weighted_Beta'] = merged_df['Weight'] * merged_df['Beta']
    portfolio_beta = merged_df['Weighted_Beta'].sum()
    
    return portfolio_beta
