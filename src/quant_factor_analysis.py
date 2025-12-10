# src/quant_factor_analysis.py
# This module contains functions for Phase 3: Quant Factor Analysis.

import pandas as pd
import numpy as np
import yfinance as yf
import yaml
import os

# --- Data Loading ---

def load_screened_tickers(data_dir, tickers_filename):
    """Loads the screened tickers from Phase 2."""
    print("Loading screened tickers...")
    filepath = os.path.join(data_dir, tickers_filename)
    with open(filepath, 'r') as f:
        return [line.strip() for line in f.readlines() if line.strip()]

def load_price_data(data_dir, prices_filename):
    """Loads the price data CSV."""
    print("Loading price data...")
    prices_path = os.path.join(data_dir, prices_filename)
    return pd.read_csv(prices_path, index_col='Date', parse_dates=True)

def fetch_benchmark_data(ticker, prices_df):
    """Fetches historical price data for the benchmark ticker."""
    print(f"Fetching benchmark data for {ticker}...")
    # Explicitly set auto_adjust=False and then select the 'Adj Close' column.
    return yf.download(ticker, start=prices_df.index.min(), end=prices_df.index.max(), auto_adjust=False)['Adj Close']

# --- Factor Calculation Helpers ---

def calculate_momentum_12_1(series):
    """Calculates 12-month momentum, excluding the last month."""
    if len(series) < 252 + 21: return np.nan
    return series.iloc[-21] / series.iloc[-252 - 21] - 1

def calculate_annualized_volatility(series):
    """Calculates annualized volatility from daily returns over the last year."""
    if len(series) < 252: return np.nan
    return series.pct_change().iloc[-252:].std() * np.sqrt(252)

def calculate_beta(stock_series, benchmark_series):
    """Calculates beta relative to a benchmark over 3 years of monthly returns."""
    stock_monthly = stock_series.resample('ME').last().pct_change().dropna()
    benchmark_monthly = benchmark_series.resample('ME').last().pct_change().dropna()
    aligned_df = pd.concat([stock_monthly, benchmark_monthly], axis=1, join='inner')
    if len(aligned_df) < 36: return np.nan
        
    stock_returns, benchmark_returns = aligned_df.iloc[-36:, 0], aligned_df.iloc[-36:, 1]
    covariance, variance = np.cov(stock_returns, benchmark_returns)[0, 1], np.var(benchmark_returns)
    
    return covariance / variance if variance != 0 else np.nan

def calculate_sharpe_ratio(series, risk_free_rate):
    """Calculates the annualized Sharpe ratio."""
    volatility = calculate_annualized_volatility(series)
    if pd.isna(volatility) or volatility == 0: return np.nan
    
    annualized_return = (1 + series.pct_change().mean()) ** 252 - 1
    return (annualized_return - risk_free_rate) / volatility

# --- Main Factor Calculation Orchestrator ---

def calculate_all_factors(tickers, prices_df, benchmark_series, risk_free_rate):
    """Calculates all quant factors for a list of tickers."""
    print("Calculating all quant factors...")
    factors = []
    for ticker in tickers:
        if ticker not in prices_df.columns: continue
        stock_prices = prices_df[ticker].dropna()
        factors.append({
            'Ticker': ticker,
            'Momentum': calculate_momentum_12_1(stock_prices),
            'Volatility': calculate_annualized_volatility(stock_prices),
            'Beta': calculate_beta(stock_prices, benchmark_series),
            'Sharpe': calculate_sharpe_ratio(stock_prices, risk_free_rate),
        })
    print("Factor calculation complete.")
    return pd.DataFrame(factors).set_index('Ticker')

# --- Scoring and Ranking ---

def rank_factors(factors_df):
    """Ranks factors using percentiles."""
    print("Ranking factors...")
    ranked_df = pd.DataFrame(index=factors_df.index)
    lower_is_better = ['Volatility', 'Beta']
    
    for factor in factors_df.columns:
        rank_col = factors_df[factor].rank(pct=True)
        if factor in lower_is_better:
            rank_col = 1 - rank_col
        ranked_df[f"{factor}_Rank"] = rank_col.fillna(0.5)
        
    print("Ranking complete.")
    return ranked_df

def calculate_quant_score(ranked_df, weights):
    """Calculates the final composite quant score."""
    print("Calculating composite quant score...")
    quant_score = pd.Series(0, index=ranked_df.index, dtype=float)
    for factor, weight in weights.items():
        rank_col = f"{factor}_Rank"
        if rank_col in ranked_df.columns:
            quant_score += ranked_df[rank_col] * weight
    
    ranked_df['Quant_Score'] = quant_score
    print("Score calculation complete.")
    return ranked_df

# --- Save Output ---

def save_quant_scores(factors_df, scored_df, data_dir, filename):
    """Saves the combined raw factors and scores to a CSV."""
    print("Saving quant scores...")
    full_df = factors_df.join(scored_df)
    filepath = os.path.join(data_dir, filename)
    full_df.sort_values(by='Quant_Score', ascending=False).to_csv(filepath)
    print(f"Quant scores saved to {filepath}")

# --- Main Phase 3 Orchestrator ---

def run_quant_factor_analysis(config_path="config.yaml"):
    """Main function to run the entire quant factor analysis phase."""
    print("\n--- Starting Phase 3: Quant Factor Analysis ---")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        data_config, screening_config, quant_config = \
            config['data_collection'], config['fundamental_screening'], config['quant_factor_analysis']

    tickers = load_screened_tickers(data_config['output_dir'], screening_config['output_tickers_filename'])
    prices = load_price_data(data_config['output_dir'], data_config['prices_filename'])
    benchmark = fetch_benchmark_data(quant_config['benchmark_ticker'], prices)

    factors_df = calculate_all_factors(tickers, prices, benchmark, quant_config['risk_free_rate'])
    ranked_df = rank_factors(factors_df)
    scored_df = calculate_quant_score(ranked_df, quant_config['score_weights'])
    
    save_quant_scores(factors_df, scored_df, data_config['output_dir'], quant_config['output_filename'])
    
    print("--- Phase 3 Finished ---")
