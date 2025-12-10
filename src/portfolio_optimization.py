# src/portfolio_optimization.py
# This module contains functions for Phase 5: Portfolio Optimization.

import pandas as pd
import numpy as np
import yaml
import os
from glob import glob
from pypfopt import EfficientFrontier, risk_models, expected_returns

# --- Data Loading ---
def load_diversified_tickers(data_dir, tickers_filename):
    """Loads the diversified tickers from Phase 4."""
    filepath = os.path.join(data_dir, tickers_filename)
    with open(filepath, 'r') as f:
        return [line.strip() for line in f.readlines() if line.strip()]

def load_price_data(data_dir, prices_filename):
    """Loads the price data CSV."""
    prices_path = os.path.join(data_dir, prices_filename)
    return pd.read_csv(prices_path, index_col='Date', parse_dates=True)

def load_latest_fundamental_data(data_dir):
    """Loads the most recent fundamentals .pkl file for market cap and sector info."""
    fundamental_files = glob(os.path.join(data_dir, 'fundamentals_*.pkl'))
    if not fundamental_files:
        raise FileNotFoundError("No fundamental data files found.")
    latest_file = max(fundamental_files, key=os.path.getctime)
    return pd.read_pickle(latest_file)

def load_optimized_weights(data_dir, weights_filename):
    """Loads the optimized portfolio weights from Phase 5."""
    filepath = os.path.join(data_dir, weights_filename)
    return pd.read_csv(filepath, index_col=0)

# --- Core Logic ---
def classify_stocks_by_market_cap(tickers, fundamentals_data, cap_threshold, etf_list):
    """Classifies stocks into large-cap and small/mid-cap groups, forcing ETFs into large-cap."""
    large_caps, small_mid_caps = [], []
    cap_groups = {}
    for ticker in tickers:
        if ticker in etf_list:
            large_caps.append(ticker)
            cap_groups[ticker] = 'Large-Cap'
            continue
        try:
            market_cap = fundamentals_data[ticker]['info'].get('marketCap')
            if market_cap is None:
                continue
            if market_cap >= cap_threshold:
                large_caps.append(ticker)
                cap_groups[ticker] = 'Large-Cap'
            else:
                small_mid_caps.append(ticker)
                cap_groups[ticker] = 'Small/Mid-Cap'
        except KeyError:
            continue
    return large_caps, small_mid_caps, cap_groups

def optimize_portfolio(tickers, prices, opt_config, always_include_tickers=None):
    """
    Optimizes a portfolio based on the specified objective function and constraints.
    Supports 'max_sharpe' and 'mean_variance' objectives.
    Always-include tickers get a minimum weight guarantee.
    """
    if not tickers:
        print("  - No tickers provided for optimization.")
        return {}
        
    objective = opt_config.get('objective', 'max_sharpe')
    min_weight_per_stock = opt_config['min_weight_per_stock']
    max_weight_per_stock = opt_config['max_weight_per_stock']
    
    print(f"  - Optimizing portfolio of {len(tickers)} tickers with objective: '{objective}'...")
    if always_include_tickers:
        print(f"  - Always-include tickers in this group: {always_include_tickers}")
    
    prices_subset = prices[tickers]
    mu = expected_returns.mean_historical_return(prices_subset)
    S = risk_models.CovarianceShrinkage(prices_subset).ledoit_wolf()
    
    # Create bounds as a list of tuples in the same order as tickers
    min_w = min_weight_per_stock if len(tickers) * min_weight_per_stock <= 1 else 0
    max_w = max_weight_per_stock if len(tickers) * max_weight_per_stock >= 1 else 1
    
    # Set minimum weight for always-include stocks (e.g., 2% minimum)
    always_include_min_weight = 0.02
    
    # Build bounds list matching the order of tickers
    bounds = []
    for ticker in tickers:
        if always_include_tickers and ticker in always_include_tickers:
            bounds.append((always_include_min_weight, max_w))
            print(f"  - Guaranteeing minimum {always_include_min_weight*100:.1f}% allocation for always-include ticker: {ticker}")
        else:
            bounds.append((min_w, max_w))
    
    ef = EfficientFrontier(mu, S, weight_bounds=bounds)
    
    try:
        if objective == 'mean_variance':
            risk_aversion = opt_config.get('risk_aversion', 1) # Default risk aversion if not specified
            print(f"  - Using mean-variance objective with risk aversion: {risk_aversion}")
            weights = ef.max_quadratic_utility(risk_aversion=risk_aversion)
        else: # Default to max_sharpe
            print("  - Using max Sharpe ratio objective.")
            weights = ef.max_sharpe()
            
        cleaned_weights = ef.clean_weights()
        
        # Verify always-include stocks are present
        if always_include_tickers:
            for ticker in always_include_tickers:
                if ticker in tickers and (ticker not in cleaned_weights or cleaned_weights[ticker] < always_include_min_weight * 0.5):
                    print(f"  - Warning: Always-include ticker {ticker} has insufficient weight. Setting to minimum.")
                    if ticker not in cleaned_weights:
                        cleaned_weights[ticker] = always_include_min_weight
                    else:
                        cleaned_weights[ticker] = max(cleaned_weights[ticker], always_include_min_weight)
                    
                    # Renormalize weights
                    total = sum(cleaned_weights.values())
                    cleaned_weights = {k: v/total for k, v in cleaned_weights.items()}
        
        if len(cleaned_weights) < len(tickers):
            print(f"  - Warning: Optimizer could not allocate to all {len(tickers)} tickers. Included {len(cleaned_weights)}.")
            
        print(f"  - Optimization successful. Allocated to {len(cleaned_weights)} tickers.")
        return cleaned_weights
    except Exception as e:
        print(f"  - Error: Portfolio optimization failed: {e}")
        # Fallback to an equal-weight portfolio if optimization fails
        print("  - Falling back to equal-weight portfolio.")
        fallback_weights = {ticker: 1/len(tickers) for ticker in tickers}
        
        # If there are always-include tickers, give them slightly higher weight
        if always_include_tickers:
            always_include_weight = 0.02
            remaining_weight = 1.0 - (len(always_include_tickers) * always_include_weight)
            other_tickers = [t for t in tickers if t not in always_include_tickers]
            
            fallback_weights = {}
            for ticker in always_include_tickers:
                if ticker in tickers:
                    fallback_weights[ticker] = always_include_weight
            
            if other_tickers:
                equal_weight = remaining_weight / len(other_tickers)
                for ticker in other_tickers:
                    fallback_weights[ticker] = equal_weight
        
        return fallback_weights

# --- Output Generation ---
def save_optimized_weights(weights, cap_groups, fundamentals_data, data_dir, filename):
    """Saves the final optimized weights to a CSV file with extra info."""
    weights_df = pd.DataFrame.from_dict(weights, orient='index', columns=['Weight'])
    weights_df['Cap_Group'] = weights_df.index.map(cap_groups)
    
    # Safely map sectors, handling missing data
    sector_map = {}
    for ticker in weights_df.index:
        if ticker in fundamentals_data and 'info' in fundamentals_data[ticker]:
            sector_map[ticker] = fundamentals_data[ticker]['info'].get('sector', 'Unknown')
        else:
            sector_map[ticker] = 'ETF'
    
    weights_df['Sector'] = weights_df.index.map(sector_map)
    filepath = os.path.join(data_dir, filename)
    weights_df.sort_values(by='Weight', ascending=False).to_csv(filepath)
    print(f"Optimized weights saved to {filepath}")

# --- Main Phase 5 Orchestrator ---
def run_portfolio_optimization(config_path="config.yaml"):
    print("\n--- Starting Phase 5: Portfolio Optimization ---")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    data_config, corr_config, opt_config, alloc_config, screening_config = \
        config['data_collection'], config['correlation_analysis'], \
        config['portfolio_optimization'], config['portfolio_allocation'], \
        config['fundamental_screening']
    
    tickers = load_diversified_tickers(data_config['output_dir'], corr_config['output_tickers_filename'])
    prices = load_price_data(data_config['output_dir'], data_config['prices_filename'])
    fundamentals = load_latest_fundamental_data(data_config['output_dir'])
    
    with open(data_config['etf_list_filename'], 'r') as f:
        etf_list = [line.strip() for line in f.readlines() if line.strip()]
    
    # Load always-include list
    from .correlation_analysis import load_always_include_list
    always_include_list = load_always_include_list(screening_config['always_include_filename'])
    
    # Filter always-include list to only those that passed price filter and are in tickers
    always_include_in_portfolio = [t for t in always_include_list if t in tickers]
    
    print(f"\n  - Always-include list from file: {always_include_list}")
    print(f"  - Always-include tickers in portfolio: {always_include_in_portfolio}")
    
    if not always_include_in_portfolio and always_include_list:
        print(f"  - WARNING: None of the always-include tickers made it to the portfolio!")
        print(f"  - Available tickers in portfolio: {tickers}")

    large_caps, small_mid_caps, cap_groups = classify_stocks_by_market_cap(tickers, fundamentals, opt_config['large_cap_threshold_usd'], etf_list)
    
    # Separate always-include stocks by cap group
    always_include_large = [t for t in always_include_in_portfolio if t in large_caps]
    always_include_small = [t for t in always_include_in_portfolio if t in small_mid_caps]
    
    print(f"  - Large-cap stocks: {large_caps}")
    print(f"  - Small/mid-cap stocks: {small_mid_caps}")
    print(f"  - Always-include in large-cap: {always_include_large}")
    print(f"  - Always-include in small/mid-cap: {always_include_small}")
    
    large_cap_alloc, small_mid_cap_alloc = alloc_config['large_cap'], alloc_config['small_mid_cap']
    if not large_caps: 
        small_mid_cap_alloc = 1.0
        large_cap_alloc = 0.0
    if not small_mid_caps: 
        large_cap_alloc = 1.0
        small_mid_cap_alloc = 0.0

    large_cap_weights = optimize_portfolio(large_caps, prices, opt_config, always_include_large)
    small_mid_cap_weights = optimize_portfolio(small_mid_caps, prices, opt_config, always_include_small)
    
    final_weights = {t: w * large_cap_alloc for t, w in large_cap_weights.items()}
    final_weights.update({t: w * small_mid_cap_alloc for t, w in small_mid_cap_weights.items()})
    
    # Final verification
    print("\n  - Verifying always-include stocks in final weights:")
    for ticker in always_include_in_portfolio:
        if ticker in final_weights:
            print(f"    ✓ {ticker}: {final_weights[ticker]*100:.2f}%")
        else:
            print(f"    ✗ {ticker}: NOT IN FINAL WEIGHTS (This should not happen!)")
    
    if not always_include_in_portfolio:
        print("    (No always-include tickers to verify)")
        
    save_optimized_weights(final_weights, cap_groups, fundamentals, data_config['output_dir'], opt_config['output_filename'])
    print("\n--- Phase 5 Finished ---")