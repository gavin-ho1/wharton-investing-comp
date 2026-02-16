# src/strategy.py
# This module encapsulates the entire investment strategy logic for a single point in time.

import pandas as pd
from .fundamental_screening import calculate_all_metrics, rank_metrics, calculate_composite_score, filter_by_price
from .quant_factor_analysis import calculate_all_factors, rank_factors, calculate_quant_score
from .correlation_analysis import calculate_hybrid_score, calculate_correlation_matrix, filter_correlated_stocks, add_always_include_stocks, load_always_include_list
from .portfolio_optimization import classify_stocks_by_market_cap, optimize_portfolio

def get_target_weights_for_date(date, all_prices, all_fundamentals, config):
    """
    Runs the entire investment strategy (Phases 2-5) for a specific date,
    ensuring no lookahead bias by using only data available up to that date.
    """
    print(f"--- Running Strategy for {date.date()} ---")
    
    prices_to_date = all_prices[all_prices.index <= date]
    fundamentals_to_date = all_fundamentals
    
    with open(config['data_collection']['etf_list_filename'], 'r') as f:
        etf_list = [line.strip() for line in f.readlines() if line.strip()]

    # --- Execute Phases ---

    # Phase 2: Price filter and Fundamental Screening
    initial_tickers = prices_to_date.columns.tolist()
    price_filtered_tickers = filter_by_price(
        prices_to_date,
        all_fundamentals,
        config['data_collection']['min_stock_price'],
        etf_list
    )

    
    etfs_to_keep = [t for t in price_filtered_tickers if t in etf_list]
    stocks_to_screen = [t for t in price_filtered_tickers if t not in etf_list]

    metrics_df = calculate_all_metrics(stocks_to_screen, fundamentals_to_date)
    ranked_metrics = rank_metrics(metrics_df)
    scored_metrics = calculate_composite_score(ranked_metrics, config['fundamental_screening']['score_weights'])
    full_scores_df = metrics_df.join(scored_metrics)
    
    cutoff = full_scores_df['Fundamental_Score'].quantile(config['fundamental_screening']['screening_percentile_cutoff'])
    screened_stocks = full_scores_df[full_scores_df['Fundamental_Score'] >= cutoff].index.tolist()
    
    screened_tickers = sorted(list(set(etfs_to_keep + screened_stocks)))
    
    # Phase 3: Quant Factor Analysis
    benchmark_ticker = config['quant_factor_analysis']['benchmark_ticker']
    benchmark_data = prices_to_date[benchmark_ticker]
    factors_df = calculate_all_factors(screened_tickers, prices_to_date, benchmark_data, config['quant_factor_analysis']['risk_free_rate'])
    ranked_factors = rank_factors(factors_df)
    quant_scores = calculate_quant_score(ranked_factors, config['quant_factor_analysis']['score_weights'])
    
    # Phase 4: Correlation & Diversification
    hybrid_scores = calculate_hybrid_score(full_scores_df, quant_scores, config['correlation_analysis']['hybrid_score_weights'])
    correlation_matrix = calculate_correlation_matrix(prices_to_date, hybrid_scores.index.tolist(), config['correlation_analysis']['lookback_years'])
    diversified_tickers, _, _, _ = filter_correlated_stocks(
        correlation_matrix, hybrid_scores, fundamentals_to_date,
        config['correlation_analysis']['correlation_threshold'],
        config['correlation_analysis']['correlation_keep_percentile'],
        etf_list
    )
    
    # Add always-include stocks AFTER all filtering
    always_include_list = load_always_include_list(config['fundamental_screening']['always_include_filename'])
    diversified_tickers = add_always_include_stocks(
        diversified_tickers,
        always_include_list,
        prices_to_date,
        fundamentals_to_date,
        config['data_collection']['min_stock_price'],
        etf_list
    )
    
    # Phase 5: Portfolio Optimization
    large_caps, small_mid_caps, _ = classify_stocks_by_market_cap(
        diversified_tickers, fundamentals_to_date, config['portfolio_optimization']['large_cap_threshold_usd'], etf_list
    )
    
    # Separate always-include stocks by cap group
    always_include_in_portfolio = [t for t in always_include_list if t in diversified_tickers]
    always_include_large = [t for t in always_include_in_portfolio if t in large_caps]
    always_include_small = [t for t in always_include_in_portfolio if t in small_mid_caps]
    
    large_cap_alloc = config['portfolio_allocation']['large_cap']
    small_mid_cap_alloc = config['portfolio_allocation']['small_mid_cap']
    if not large_caps:
        small_mid_cap_alloc = 1.0
        large_cap_alloc = 0.0
    if not small_mid_caps:
        large_cap_alloc = 1.0
        small_mid_cap_alloc = 0.0

    opt_config = config['portfolio_optimization']
    large_cap_weights = optimize_portfolio(large_caps, prices_to_date, opt_config, always_include_large)
    small_mid_cap_weights = optimize_portfolio(small_mid_caps, prices_to_date, opt_config, always_include_small)
    
    final_weights = {}
    for ticker, weight in large_cap_weights.items():
        final_weights[ticker] = weight * large_cap_alloc
    for ticker, weight in small_mid_cap_weights.items():
        final_weights[ticker] = weight * small_mid_cap_alloc
        
    return final_weights