# src/projection.py
# This module contains functions for Phase 6: Forward-Looking Monte Carlo Projection.

import pandas as pd
import numpy as np
import yaml
import os
import matplotlib.pyplot as plt
import quantstats as qs
import yfinance as yf
from src.simulation_models import run_gbm_simulation, run_heston_merton_simulation, run_multi_asset_heston_merton

def run_monte_carlo_simulation(weights, prices, config):
    """
    Orchestrates the Monte Carlo simulation by selecting the appropriate model 
    based on the configuration. Uses batch processing to reduce memory usage.
    """
    proj_config = config['projection']
    model_config = proj_config['simulation_model']
    
    # --- 1. Prepare Data and Assumptions ---
    print("Preparing data for Monte Carlo simulation...")
    
    lookback_years = proj_config['lookback_years']
    end_date = prices.index.max()
    start_date = end_date - pd.DateOffset(years=lookback_years)
    hist_prices = prices[start_date:end_date]
    
    daily_returns = hist_prices[weights.index].pct_change().dropna()
    initial_weights = weights['Weight'].values
    initial_investment = proj_config['initial_investment']

    # --- Target Volatility Scaling ---
    if proj_config.get('enable_target_vol_scaling', False):
        print("  - Target volatility scaling is enabled.")
        target_vol = proj_config.get('target_volatility', 0.15)
        
        # Calculate annualized historical portfolio volatility
        cov_matrix = daily_returns.cov() * 252 # Annualize covariance
        port_var = initial_weights.T @ cov_matrix @ initial_weights
        port_vol = np.sqrt(port_var)
        
        if port_vol > 0:
            alpha = target_vol / port_vol
            alpha = min(alpha, 1.0) # Do not increase exposure, only scale down
            
            print(f"  - Historical annualized volatility: {port_vol:.2%}")
            print(f"  - Target volatility: {target_vol:.2%}")
            print(f"  - Scaling factor (alpha): {alpha:.4f}")
            
            scaled_weights = initial_weights * alpha
            
            # Renormalize weights to sum to 1 to remain fully invested
            initial_weights = scaled_weights / scaled_weights.sum()
            
            print(f"  - New scaled and renormalized weights sum: {initial_weights.sum():.2%}")
        else:
            print("  - Warning: Portfolio volatility is zero. Skipping scaling.")


    # --- 2. Batch Processing Setup ---
    n_simulations = proj_config['n_simulations']
    batch_size = proj_config.get('batch_size', 500)
    n_batches = int(np.ceil(n_simulations / batch_size))
    
    print(f"\n--- Batch Processing: {n_simulations} simulations in {n_batches} batches of {batch_size} ---")
    
    # --- 3. Select Simulation Model and Run in Batches ---
    sim_level = model_config.get('simulation_level', 'portfolio')
    use_sv = model_config['use_stochastic_vol']
    use_jumps = model_config['use_jump_diffusion']
    
    # Determine which simulation function to use
    if not use_sv and not use_jumps:
        print("Model selected: Geometric Brownian Motion (GBM) - Multi-Asset")
        sim_function = lambda batch_config: run_gbm_simulation(daily_returns, batch_config, initial_investment, initial_weights)
    elif sim_level == 'asset':
        print("Model selected: Heston-Merton - Asset-Level")
        benchmark_ticker = config['quant_factor_analysis']['benchmark_ticker']
        benchmark_returns = hist_prices[benchmark_ticker].pct_change().dropna()
        sim_function = lambda batch_config: run_multi_asset_heston_merton(daily_returns, benchmark_returns, batch_config, initial_investment, initial_weights)
    else:
        print("Model selected: Heston-Merton - Portfolio-Level")
        portfolio_daily_returns = (daily_returns * initial_weights).sum(axis=1)
        sim_function = lambda batch_config: run_heston_merton_simulation(portfolio_daily_returns, batch_config, initial_investment)
    
    # --- 4. Run Batches and Aggregate Results ---
    all_portfolio_paths = []
    all_vol_paths = []
    
    for batch_idx in range(n_batches):
        # Calculate batch size for this iteration
        start_sim = batch_idx * batch_size
        end_sim = min(start_sim + batch_size, n_simulations)
        current_batch_size = end_sim - start_sim
        
        print(f"\nRunning batch {batch_idx + 1}/{n_batches} ({current_batch_size} simulations)...")
        
        # Create a modified config for this batch
        batch_config = config.copy()
        batch_config['projection'] = proj_config.copy()
        batch_config['projection']['n_simulations'] = current_batch_size
        
        # Run the simulation for this batch
        batch_portfolio_paths, batch_vol_paths = sim_function(batch_config)
        
        # Store results
        all_portfolio_paths.append(batch_portfolio_paths)
        if batch_vol_paths is not None:
            all_vol_paths.append(batch_vol_paths)
    
    # --- 5. Concatenate All Batches ---
    print("\nCombining batch results...")
    portfolio_paths = np.concatenate(all_portfolio_paths, axis=1)
    vol_paths = np.concatenate(all_vol_paths, axis=1) if all_vol_paths else None
    
    print(f"Simulation complete. Total simulations: {portfolio_paths.shape[1]}")
    return portfolio_paths, vol_paths

def analyze_and_save_projections(portfolio_paths, volatility_paths, config, prices):
    """
    Analyzes simulation results and saves all outputs for Phase 6, with enhanced, consistent plotting.
    """
    import os
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    import numpy as np
    import pandas as pd

    print("Analyzing and saving projection results with enhanced visualizations...")

    # --- 1. Fetch Benchmark Data ---
    benchmark_ticker = config['quant_factor_analysis']['benchmark_ticker']
    horizon_years = config['projection']['horizon_years']
    end_date = prices.index.max()
    start_date = end_date - pd.DateOffset(years=horizon_years)
    
    print(f"Fetching {horizon_years} years of historical data for benchmark: {benchmark_ticker}...")
    benchmark_prices = yf.download(benchmark_ticker, start=start_date, end=end_date, progress=False)['Close']

    # --- 2. Calculate Statistics ---
    initial_investment = config['projection']['initial_investment']
    proj_config = config['projection']
    data_dir = config['data_collection']['output_dir']
    risk_free_rate = config['quant_factor_analysis']['risk_free_rate']
    horizon_years = proj_config['horizon_years']
    final_values = portfolio_paths[-1]

    # --- 3. Prepare Summary Table ---
    summary_data = []
    percentile_values = {}
    for percentile in proj_config['plot_percentiles']:
        final_value = np.percentile(final_values, percentile)
        percentile_values[percentile] = final_value
        cagr = ((final_value / initial_investment) ** (1 / horizon_years)) - 1
        
        percentile_path_idx = np.argmin(np.abs(final_values - final_value))
        returns = pd.Series(portfolio_paths[:, percentile_path_idx]).pct_change().dropna()
        annual_volatility = returns.std() * np.sqrt(12)
        sharpe_ratio = (cagr - risk_free_rate) / annual_volatility if annual_volatility > 0 else 0

        summary_data.append({
            'Percentile': f"{percentile}th", 'Final Portfolio Value': f"${final_value:,.2f}",
            'CAGR': f"{cagr:.2%}", 'Annualized Volatility': f"{annual_volatility:.2%}",
            'Sharpe Ratio': f"{sharpe_ratio:.2f}"
        })

    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(data_dir, proj_config['summary_filename'])
    summary_df.to_csv(summary_path, index=False)
    print(f"Projection summary saved to {summary_path}")
    print("\n" + summary_df.to_string())

    reports_dir = 'reports'
    os.makedirs(reports_dir, exist_ok=True)

    # --- 4. Plotting Setup ---
    import matplotlib.ticker as mticker
    cmap = plt.cm.magma
    n_paths = portfolio_paths.shape[1]
    time_steps = np.arange(len(portfolio_paths))
    percentile_ranks = np.argsort(np.argsort(final_values)) / (n_paths - 1)
    path_colors = [cmap(rank) for rank in percentile_ranks]
    highlight_percentiles = [10, 50, 90]
    highlight_colors = {10: '#8B0000', 50: '#000000', 90: '#006400'}

    def val_to_pct_format(y, pos):
        return f'{(y / initial_investment - 1) * 100:,.0f}%'
        
    def setup_log_axis(ax):
        ax.set_yscale('log')
        # Use LogLocator for major ticks
        locmin = mticker.LogLocator(base=10.0, subs=np.arange(0.1,1,0.1), numticks=12)
        ax.yaxis.set_minor_locator(locmin)
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        ax.grid(True, which="major", ls="--", linewidth=0.7)
        ax.grid(True, which="minor", ls=":", linewidth=0.5)

    # --- 5. Main Tearsheet Plot ---
    fig = plt.figure(figsize=(20, 18), constrained_layout=True)
    fig.suptitle('Monte Carlo Projection Analysis', fontsize=24, weight='bold')
    gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1], height_ratios=[1.5, 1])
    
    # Create subplots
    ax_sim_paths = fig.add_subplot(gs[0, 0])
    ax_hist = fig.add_subplot(gs[0, 1])
    ax_table_pct = fig.add_subplot(gs[1, 0])
    ax_table_comp = fig.add_subplot(gs[1, 1])

    # --- Plot 1: Simulation Paths (Top Left) ---
    num_sample_paths = min(n_paths, 100)
    sample_indices = np.random.choice(n_paths, num_sample_paths, replace=False)
    for i in sample_indices:
        ax_sim_paths.plot(time_steps, portfolio_paths[:, i], color=path_colors[i], alpha=0.6, linewidth=0.8)

    for percentile in proj_config['plot_percentiles']:
        line = np.percentile(portfolio_paths, percentile, axis=1)
        color = highlight_colors.get(percentile, cmap(percentile / 100))
        ax_sim_paths.plot(time_steps, line, linewidth=2, alpha=1, color=color, label=f"{percentile}th Percentile")
        final_val = percentile_values[percentile]
        label_text = f"{percentile}th: ${final_val/1e6:.2f}M" if final_val > 1e6 else f"{percentile}th: ${final_val/1e3:.2f}K"
        ax_sim_paths.annotate(label_text, xy=(len(portfolio_paths)-1, line[-1]),
                     xytext=(5, 0), textcoords='offset points', va='center', ha='left', fontsize=10, color=color)

    setup_log_axis(ax_sim_paths)
    ax_sim_paths.set_ylabel('Portfolio Value ($, log scale)')
    ax_sim_paths.set_xlabel(f'Months (Years: {horizon_years})')
    ax_sim_paths.set_title(f'Sample of {num_sample_paths} Monte Carlo Simulations')
    ax_sim_paths.legend()

    ax_pct_twin = ax_sim_paths.twinx()
    ax_pct_twin.set_yscale('log')
    ax_pct_twin.set_ylim(ax_sim_paths.get_ylim())
    ax_pct_twin.yaxis.set_major_formatter(FuncFormatter(val_to_pct_format))
    ax_pct_twin.set_ylabel('Portfolio Growth (%)')
    main_plot_ylim = ax_sim_paths.get_ylim() # Save for consistent scaling
    
    # --- Plot 2: Histogram of Final Values (Top Right) ---
    upper_bound = np.percentile(final_values, 99)
    bounded_values = final_values[final_values <= upper_bound]
    
    # Use Freedman-Diaconis rule for binning
    q25, q75 = np.percentile(bounded_values, [25, 75])
    iqr = q75 - q25
    bin_width = 2 * iqr * len(bounded_values) ** (-1/3) if iqr > 0 else 0
    bins = int(np.ceil((bounded_values.max() - bounded_values.min()) / bin_width)) if bin_width > 0 else 50
    
    ax_hist.hist(bounded_values, bins=bins, edgecolor='black', alpha=0.75, color='skyblue')
    ax_hist.set_xlim(0, upper_bound)

    for percentile in highlight_percentiles:
        val = percentile_values[percentile]
        if val <= upper_bound:
            label = f"{percentile}th: ${val/1e6:.2f}M" if val > 1e6 else f"{percentile}th: ${val/1e3:.2f}K"
            ax_hist.axvline(val, color=highlight_colors[percentile], linestyle='dashed', linewidth=2, label=label)

    ax_hist.set_title('Distribution of Final Values (Capped at 99th pctl)')
    ax_hist.set_xlabel('Final Value ($)')
    ax_hist.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
    ax_hist.legend()

    # --- Table 1: Percentile Statistics (Bottom Left) ---
    ax_table_pct.axis('off') # Hide the axes
    ax_table_pct.set_title('Simulation Outcome Statistics (by Percentile)', pad=20)
    table = ax_table_pct.table(
        cellText=summary_df.values,
        colLabels=summary_df.columns,
        loc='center',
        cellLoc='center',
        bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.scale(1, 1.5)
    
    # --- Table 2: Comparison vs. Benchmark (Bottom Right) ---
    ax_table_comp.axis('off')
    ax_table_comp.set_title(f'Mean Simulation vs. {benchmark_ticker} ({horizon_years} Yr Equivalent)', pad=20)

    # Calculate mean simulation statistics by averaging metrics from each individual path
    path_cagrs = []
    path_vols = []
    path_sharpes = []

    for i in range(portfolio_paths.shape[1]):
        path = portfolio_paths[:, i]
        final_value = path[-1]
        
        # Calculate CAGR for the path
        cagr = ((final_value / initial_investment) ** (1 / horizon_years)) - 1
        path_cagrs.append(cagr)
        
        # Calculate Volatility for the path
        returns = pd.Series(path).pct_change().dropna()
        if not returns.empty:
            annual_volatility = returns.std() * np.sqrt(12) # Monthly simulation
            path_vols.append(annual_volatility)
            
            # Calculate Sharpe Ratio for the path
            sharpe_ratio = (cagr - risk_free_rate) / annual_volatility if annual_volatility > 0 else 0
            path_sharpes.append(sharpe_ratio)
        else:
            # Handle cases with no returns (e.g., constant value path)
            path_vols.append(0)
            path_sharpes.append(0)

    mean_cagr = np.mean(path_cagrs)
    mean_annual_volatility = np.mean(path_vols)
    mean_sharpe = np.mean(path_sharpes)
    
    # Total return is simpler to calculate from the average final value
    mean_final_value = final_values.mean()
    mean_total_return = (mean_final_value / initial_investment) - 1

    # Calculate historical benchmark stats
    # --- Determine lookback window for benchmark Sharpe & CAGR ---
    projection_cfg = config["projection"]
    lookback_years = projection_cfg.get("sharpe_lookback_years", projection_cfg["horizon_years"])

    end_date = pd.Timestamp.today()
    start_date = end_date - pd.DateOffset(years=lookback_years)

    # Filter benchmark data to match lookback window
    benchmark_prices_window = benchmark_prices.loc[start_date:end_date]
    benchmark_monthly_prices = benchmark_prices_window.resample('ME').last()
    benchmark_monthly_returns = benchmark_monthly_prices.pct_change().dropna()

    # --- Compute benchmark metrics (CAGR, Volatility, Sharpe, Total Return) ---
    spy_cagr = qs.stats.cagr(benchmark_monthly_returns, periods=12).item()
    spy_volatility = qs.stats.volatility(benchmark_monthly_returns, periods=12).item()
    spy_sharpe = qs.stats.sharpe(
        benchmark_monthly_returns,
        rf=risk_free_rate,
        periods=12
    ).item()

    # Ensure native float to avoid numpy float issues
    spy_total_return = float((benchmark_monthly_prices.iloc[-1] / benchmark_monthly_prices.iloc[0]) - 1)

    # Create and format comparison table
    comp_data = {
        'Metric': ['CAGR', 'Annualized Volatility', 'Sharpe Ratio', 'Total Return'],
        'Mean Simulation': [f"{mean_cagr:.2%}", f"{mean_annual_volatility:.2%}", f"{mean_sharpe:.2f}", f"{mean_total_return:.2%}"],
        f'{benchmark_ticker} Equivalent': [f"{spy_cagr:.2%}", f"{spy_volatility:.2%}", f"{spy_sharpe:.2f}", f"{spy_total_return:.2%}"]
    }
    comp_df = pd.DataFrame(comp_data)

    comp_table = ax_table_comp.table(
        cellText=comp_df.values,
        colLabels=comp_df.columns,
        loc='center',
        cellLoc='center',
        bbox=[0, 0, 1, 1]
    )
    comp_table.auto_set_font_size(False)
    comp_table.scale(1, 1.5)


    plt.savefig(os.path.join(reports_dir, proj_config['main_plot_filename']), dpi=300)

    # --- 6. Generate Standalone Histogram Plot ---
    fig_hist, ax_hist_standalone = plt.subplots(figsize=(14, 6))
    ax_hist_standalone.hist(bounded_values, bins=bins, edgecolor='black', alpha=0.75, color='skyblue')
    ax_hist_standalone.set_xlim(0, upper_bound)
    for percentile in highlight_percentiles:
        val = percentile_values[percentile]
        if val <= upper_bound:
            label = f"{percentile}th: ${val/1e6:.2f}M" if val > 1e6 else f"{percentile}th: ${val/1e3:.2f}K"
            ax_hist_standalone.axvline(val, color=highlight_colors[percentile], linestyle='dashed', linewidth=2, label=label)
    ax_hist_standalone.set_title('Distribution of Final Portfolio Values (capped at 99th percentile)')
    ax_hist_standalone.set_xlabel('Final Value ($)')
    ax_hist_standalone.legend()
    fig_hist.savefig(os.path.join(reports_dir, proj_config['histogram_plot_filename']), dpi=300)
    plt.close(fig_hist)

    # --- 7. Generate Standalone Simulation Path Plots ---
    
    # Plot 1: Sample Paths (for simulation_paths.png)
    fig_sim_sample, ax_sim_sample = plt.subplots(figsize=(16, 6))
    for i in sample_indices: # Use the same sample as the tearsheet
        ax_sim_sample.plot(time_steps, portfolio_paths[:, i], color=path_colors[i], alpha=0.6, linewidth=0.8)

    for percentile in proj_config['plot_percentiles']:
        line = np.percentile(portfolio_paths, percentile, axis=1)
        color = highlight_colors.get(percentile, cmap(percentile / 100))
        ax_sim_sample.plot(time_steps, line, linewidth=2.5, alpha=1, color=color, label=f"{percentile}th Percentile")

    setup_log_axis(ax_sim_sample)
    ax_sim_sample.set_ylim(main_plot_ylim)
    ax_sim_sample.set_title(f'Monte Carlo Simulation Paths (Sample of {num_sample_paths})')
    ax_sim_sample.set_xlabel('Months')
    ax_sim_sample.set_ylabel('Portfolio Value ($, log scale)')
    ax_sim_sample.legend()

    ax_sim_pct_sample = ax_sim_sample.twinx()
    setup_log_axis(ax_sim_pct_sample)
    ax_sim_pct_sample.set_ylim(main_plot_ylim)
    ax_sim_pct_sample.yaxis.set_major_formatter(FuncFormatter(val_to_pct_format))
    ax_sim_pct_sample.set_ylabel('Portfolio Growth (%)')

    fig_sim_sample.savefig(os.path.join(reports_dir, proj_config['simulation_paths_plot_filename']), dpi=300)
    plt.close(fig_sim_sample)

    # Plot 2: All Paths (for simulation_paths_full.png)
    fig_sim_full, ax_sim_full = plt.subplots(figsize=(16, 6))
    for i in range(n_paths):
        ax_sim_full.plot(time_steps, portfolio_paths[:, i], color=path_colors[i], alpha=0.5, linewidth=0.8)

    for percentile in proj_config['plot_percentiles']:
        line = np.percentile(portfolio_paths, percentile, axis=1)
        color = highlight_colors.get(percentile, cmap(percentile / 100))
        ax_sim_full.plot(time_steps, line, linewidth=2.5, alpha=1, color=color)

    setup_log_axis(ax_sim_full)
    ax_sim_full.set_ylim(main_plot_ylim)
    ax_sim_full.set_title('Monte Carlo Simulation Paths (All)')
    ax_sim_full.set_xlabel('Months')
    ax_sim_full.set_ylabel('Portfolio Value ($, log scale)')

    ax_sim_pct_full = ax_sim_full.twinx()
    setup_log_axis(ax_sim_pct_full)
    ax_sim_pct_full.set_ylim(main_plot_ylim)
    ax_sim_pct_full.yaxis.set_major_formatter(FuncFormatter(val_to_pct_format))
    ax_sim_pct_full.set_ylabel('Portfolio Growth (%)')

    # Define a new filename in config or hardcode here
    full_plot_filename = proj_config['simulation_paths_plot_filename'].replace('.png', '_full.png')
    fig_sim_full.savefig(os.path.join(reports_dir, full_plot_filename), dpi=300)
    plt.close(fig_sim_full)


# --- Main Phase 6 Orchestrator ---

def run_projection(config_path="config.yaml"):
    """Main function to run the entire projection phase."""
    print("\n--- Starting Phase 6: Monte Carlo Projection ---")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    data_dir = config['data_collection']['output_dir']
    
    weights_path = os.path.join(data_dir, config['portfolio_optimization']['output_filename'])
    prices_path = os.path.join(data_dir, config['data_collection']['prices_filename'])
    
    weights = pd.read_csv(weights_path, index_col=0)
    prices = pd.read_csv(prices_path, index_col='Date', parse_dates=True)
    
    starting_weights_path = os.path.join(data_dir, config['projection']['starting_weights_filename'])
    weights.sort_values(by='Weight', ascending=False).to_csv(starting_weights_path)
    print(f"Projection starting weights saved to {starting_weights_path}")

    portfolio_paths, volatility_paths = run_monte_carlo_simulation(weights, prices, config)
    
    analyze_and_save_projections(portfolio_paths, volatility_paths, config, prices)
    
    print("\n--- Phase 6 Finished ---")
