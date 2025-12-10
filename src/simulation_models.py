# src/simulation_models.py
# This module contains different stochastic models for Monte Carlo simulation.

import numpy as np

def _get_positive_definite_matrix(matrix, epsilon=1e-6, max_tries=10):
    """
    Ensures a matrix is positive definite by adding a small identity matrix.
    """
    if matrix is None:
        return None
    
    matrix = np.array(matrix)
    identity = np.eye(matrix.shape[0])
    tries = 0
    
    while tries < max_tries:
        try:
            # Try to compute the Cholesky decomposition
            np.linalg.cholesky(matrix)
            # If it succeeds, the matrix is positive definite
            return matrix
        except np.linalg.LinAlgError:
            # If it fails, add a small epsilon to the diagonal
            matrix = matrix + epsilon * identity
            tries += 1
            
    # If it still fails after max_tries, raise an error
    raise np.linalg.LinAlgError("Matrix could not be made positive definite")

def _apply_weight_constraints(weights, max_weight):
    """
    Applies a maximum weight constraint to a weights vector, redistributing excess weight.
    This is a single-pass redistribution, which is fast and robust.
    """
    # If constraints are already met, just normalize and return.
    if np.all(weights <= max_weight):
        return weights / weights.sum()

    # 1. Cap weights at max_weight
    capped_weights = np.minimum(weights, max_weight)
    
    # 2. Calculate the total excess weight from all assets that were capped
    excess_weight = np.sum(weights) - np.sum(capped_weights)

    # 3. Identify assets eligible to receive the redistributed weight
    eligible_mask = capped_weights < max_weight
    
    # If no assets are eligible (an unlikely edge case), we can't redistribute.
    # Just return the capped weights, normalized.
    if not np.any(eligible_mask):
        return capped_weights / np.sum(capped_weights)

    # 4. Get the current weights of eligible assets to determine redistribution proportion
    eligible_weights = capped_weights[eligible_mask]
    
    # 5. Distribute the excess weight. If all eligible weights are 0, distribute equally.
    if eligible_weights.sum() > 1e-6:
        redistribution = excess_weight * (eligible_weights / eligible_weights.sum())
    else:
        redistribution = excess_weight / len(eligible_weights)
        
    capped_weights[eligible_mask] += redistribution
    
    # 6. Final normalization to correct any minor floating point errors.
    return capped_weights / np.sum(capped_weights)

def _get_dynamic_weights(monthly_returns, fundamental_weights, alpha, max_weight):
    """
    Calculates dynamic weights based on a blend of fundamental scores and price momentum.
    """
    # 1. Calculate price-based (momentum) weights from last month's returns
    price_weights = np.maximum(monthly_returns, 0)
    
    # 2. Handle edge case: if all returns are negative, use equal weight for the momentum part
    if price_weights.sum() < 1e-6:
        price_weights = np.ones_like(price_weights) / len(price_weights)
    else:
        price_weights /= price_weights.sum()
        
    # 3. Blend fundamental and price-based weights using alpha
    new_weights = alpha * fundamental_weights + (1 - alpha) * price_weights
    
    # 4. Apply constraints (e.g., max weight per stock) and normalize
    constrained_weights = _apply_weight_constraints(new_weights, max_weight)
    
    return constrained_weights

def run_gbm_simulation(hist_returns, config, initial_investment, initial_weights):
    """
    Runs a multi-asset Monte Carlo simulation using Geometric Brownian Motion.
    This model uses MONTHLY time steps and supports dynamic rebalancing.
    """
    proj_config = config['projection']
    opt_config = config['portfolio_optimization']
    
    # --- 1. Get Parameters ---
    return_shrinkage = proj_config.get('return_shrinkage', 1.0)
    mean_returns = hist_returns.mean() * 252 * return_shrinkage
    if return_shrinkage != 1.0:
        print(f"  - Applying return shrinkage factor: {return_shrinkage}")
    cov_matrix = hist_returns.cov() * 252
    # Handle NaN values that can arise from missing data
    cov_matrix = np.nan_to_num(cov_matrix, nan=0.0)
    
    n_simulations = proj_config['n_simulations']
    horizon_years = proj_config['horizon_years']
    n_months = horizon_years * 12
    dt = 1/12
    
    # Rebalancing settings
    enable_monthly_rebalancing = proj_config.get('enable_monthly_rebalancing', False)
    alpha = proj_config.get('rebalancing_alpha', 0.7)
    max_weight = opt_config.get('max_weight_per_stock', 0.1)
    
    # --- 2. Initialize Arrays ---
    portfolio_values = np.zeros((n_months + 1, n_simulations))
    portfolio_values[0] = initial_investment
    
    # Store last month's returns for each simulation to calculate momentum
    last_month_returns = np.zeros((len(initial_weights), n_simulations))

    print(f"Running {n_simulations} GBM simulations for {horizon_years} years (Monthly Steps)...")
    
    # --- 3. Run Simulation Loop ---
    current_weights = np.tile(initial_weights, (n_simulations, 1)).T
    
    for t in range(1, n_months + 1):
        # --- Rebalance at the start of the month ---
        if enable_monthly_rebalancing:
            for i in range(n_simulations):
                current_weights[:, i] = _get_dynamic_weights(last_month_returns[:, i], initial_weights, alpha, max_weight)
        elif proj_config['rebalance_annually'] and t % 12 == 0:
            current_weights = np.tile(initial_weights, (n_simulations, 1)).T

        # --- Simulate this month's returns ---
        pos_def_cov_matrix = _get_positive_definite_matrix(cov_matrix)
        L = np.linalg.cholesky(pos_def_cov_matrix)
        z = np.random.normal(0, 1, size=(len(initial_weights), n_simulations))
        monthly_asset_returns = mean_returns[:, np.newaxis] * dt + (L @ z) * np.sqrt(dt)
        last_month_returns = monthly_asset_returns # Save for next iteration
        
        # --- Update portfolio value ---
        portfolio_return = np.sum(current_weights * monthly_asset_returns, axis=0)
        portfolio_values[t, :] = portfolio_values[t-1, :] * (1 + portfolio_return)
        
        # --- Update weights for drift (if not rebalancing) ---
        if not (enable_monthly_rebalancing or (proj_config['rebalance_annually'] and (t+1) % 12 == 0)):
            asset_values = current_weights * (1 + monthly_asset_returns)
            current_weights = asset_values / np.sum(asset_values, axis=0)
            
    print("GBM simulation complete.")
    return portfolio_values, None

def run_multi_asset_heston_merton(hist_returns, benchmark_returns, config, initial_investment, initial_weights):
    """
    Runs a multi-asset Heston-Merton simulation with a monthly rebalancing loop.
    """
    # --- 1. Load Configuration ---
    proj_config = config['projection']
    opt_config = config['portfolio_optimization']
    model_config = proj_config['simulation_model']
    heston_params = model_config['heston_params']
    asset_level_params = model_config['asset_level_params']
    sys_jump_params = asset_level_params['systemic_jumps']
    idio_jump_params = asset_level_params['idiosyncratic_jumps']

    # Rebalancing settings
    enable_monthly_rebalancing = proj_config.get('enable_monthly_rebalancing', False)
    alpha = proj_config.get('rebalancing_alpha', 0.7)
    max_weight = opt_config.get('max_weight_per_stock', 0.1)
    
    # --- 2. Set Up Simulation Parameters ---
    n_assets = len(initial_weights)
    n_simulations = proj_config['n_simulations']
    horizon_years = proj_config['horizon_years']
    n_months = horizon_years * 12
    days_per_month = 21
    n_steps = n_months * days_per_month
    dt = 1 / 252

    print(f"Running {n_simulations} multi-asset Heston-Merton simulations for {horizon_years} years...")

    # --- 3. Prepare Financial Data ---
    return_shrinkage = proj_config.get('return_shrinkage', 1.0)
    mean_returns = hist_returns.mean().values * 252 * return_shrinkage
    if return_shrinkage != 1.0:
        print(f"  - Applying return shrinkage factor: {return_shrinkage}")
    corr_matrix = hist_returns.corr().values
    # Handle NaN values that can arise from missing data
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    corr_matrix = _get_positive_definite_matrix(corr_matrix)
    L = np.linalg.cholesky(corr_matrix)
    asset_vols = hist_returns.std().values * np.sqrt(252)
    market_vol = benchmark_returns.std() * np.sqrt(252)
    alpha_vol = asset_vols / market_vol if market_vol > 0 else np.ones(n_assets)

    # --- 4. Set Up Model Parameters ---
    v0_market = benchmark_returns.var() * 252
    kappa, theta, sigma_v, rho = heston_params['kappa'], heston_params['theta'], heston_params['sigma_v'], heston_params['rho']
    lambda_sys, mu_sys, sigma_sys = sys_jump_params['lambda_sys'], sys_jump_params['mu_sys'], sys_jump_params['sigma_sys']
    lambda_id, mu_id, sigma_id = idio_jump_params['lambda_id'], idio_jump_params['mu_id'], idio_jump_params['sigma_id']
    k_sys = np.exp(mu_sys + 0.5 * sigma_sys**2) - 1
    k_id = np.exp(mu_id + 0.5 * sigma_id**2) - 1

    # --- 5. Generate Random Numbers ---
    Z_v = np.random.normal(size=(n_steps, n_simulations))
    Z_s_ind = np.random.normal(size=(n_steps, n_assets, n_simulations))
    Z_s_corr = np.einsum('ij,kjl->kil', L, Z_s_ind)
    Z_s = rho * Z_v[:, np.newaxis, :] + np.sqrt(1 - rho**2) * Z_s_corr
    dW_v = np.sqrt(dt) * Z_v
    dW_s = np.sqrt(dt) * Z_s
    poisson_sys = np.random.poisson(lambda_sys * dt, size=(n_steps, n_simulations))
    poisson_id = np.random.poisson(lambda_id * dt, size=(n_steps, n_assets, n_simulations))

    # --- 6. Run Vectorized Daily Price Simulation (as relative price paths) ---
    asset_prices = np.ones((n_steps + 1, n_assets, n_simulations))
    market_variance = np.zeros((n_steps + 1, n_simulations))
    market_variance[0, :] = v0_market

    for i in range(n_steps):
        v_t = np.maximum(market_variance[i, :], 0)
        dv = kappa * (theta - v_t) * dt + sigma_v * np.sqrt(v_t) * dW_v[i, :]
        market_variance[i+1, :] = market_variance[i, :] + dv
        current_asset_variance = (alpha_vol[:, np.newaxis]**2) * v_t
        sys_jump_factor = np.power(np.exp(np.random.normal(mu_sys, sigma_sys, size=n_simulations)), (poisson_sys[i, :] > 0))
        idio_jump_factor = np.power(np.exp(np.random.normal(mu_id, sigma_id, size=(n_assets, n_simulations))), (poisson_id[i, :, :] > 0))
        drift = (mean_returns[:, np.newaxis] - 0.5 * current_asset_variance - (lambda_sys * k_sys) - (lambda_id * k_id)) * dt
        diffusion = np.sqrt(current_asset_variance) * dW_s[i, :, :]
        asset_prices[i+1, :, :] = asset_prices[i, :, :] * np.exp(drift + diffusion) * sys_jump_factor[np.newaxis, :] * idio_jump_factor

    # --- 7. Calculate Portfolio Value with Monthly Rebalancing ---
    portfolio_values = np.zeros((n_months + 1, n_simulations))
    portfolio_values[0, :] = initial_investment
    current_weights = np.tile(initial_weights, (n_simulations, 1)).T
    
    for t in range(1, n_months + 1):
        # --- Rebalance at the start of the month ---
        if enable_monthly_rebalancing and t > 1:
            prev_month_start_idx = (t - 2) * days_per_month
            prev_month_end_idx = (t - 1) * days_per_month
            monthly_asset_returns = (asset_prices[prev_month_end_idx] / asset_prices[prev_month_start_idx]) - 1
            for i in range(n_simulations):
                current_weights[:, i] = _get_dynamic_weights(monthly_asset_returns[:, i], initial_weights, alpha, max_weight)
        elif proj_config['rebalance_annually'] and t > 1 and t % 12 == 0:
            current_weights = np.tile(initial_weights, (n_simulations, 1)).T
        
        # --- Calculate portfolio performance for the current month ---
        current_month_start_idx = (t - 1) * days_per_month
        current_month_end_idx = t * days_per_month
        month_asset_returns = (asset_prices[current_month_end_idx] / asset_prices[current_month_start_idx]) - 1
        monthly_portfolio_return = np.sum(current_weights * month_asset_returns, axis=0)
        portfolio_values[t, :] = portfolio_values[t-1, :] * (1 + monthly_portfolio_return)
        
        # --- Update weights for drift if not rebalancing next period ---
        if not enable_monthly_rebalancing and not (proj_config['rebalance_annually'] and (t+1) % 12 == 0):
            asset_values = current_weights * (1 + month_asset_returns)
            current_weights = asset_values / np.sum(asset_values, axis=0, keepdims=True)

    # --- 8. Downsample Volatility to Monthly for Reporting ---
    monthly_indices = np.linspace(0, n_steps, n_months + 1, dtype=int)
    monthly_vol_paths = market_variance[monthly_indices, :]
    
    print("Multi-asset Heston-Merton simulation complete.")
    return portfolio_values, monthly_vol_paths

def run_heston_merton_simulation(hist_returns, config, initial_investment):
    """
    Runs a portfolio-level Monte Carlo simulation using a combined Heston (stochastic volatility)
    and Merton (jump diffusion) model. This model uses DAILY time steps.
    """
    proj_config = config['projection']
    model_config = proj_config['simulation_model']
    heston_params = model_config['heston_params']
    merton_params = model_config['merton_params']
    
    use_sv = model_config['use_stochastic_vol']
    use_jumps = model_config['use_jump_diffusion']
    
    return_shrinkage = proj_config.get('return_shrinkage', 1.0)
    mu = hist_returns.mean() * 252 * return_shrinkage
    if return_shrinkage != 1.0:
        print(f"  - Applying return shrinkage factor: {return_shrinkage}")
    v0 = hist_returns.var() * 252
    kappa, theta, sigma_v, rho = heston_params['kappa'], heston_params['theta'], heston_params['sigma_v'], heston_params['rho']
    lambda_j, mu_j, sigma_j = merton_params['lambda'], merton_params['mu_j'], merton_params['sigma_j']
    
    n_simulations = proj_config['n_simulations']
    horizon_years = proj_config['horizon_years']
    n_steps = horizon_years * 252
    dt = 1 / 252

    portfolio_values = np.zeros((n_steps + 1, n_simulations))
    portfolio_values[0] = initial_investment
    variance_paths = np.zeros((n_steps + 1, n_simulations))
    variance_paths[0] = v0
    k = np.exp(mu_j + 0.5 * sigma_j**2) - 1
    
    model_desc = [name for name, flag in [("Heston", use_sv), ("Merton", use_jumps)] if flag]
    print(f"Running {n_simulations} {'-'.join(model_desc)} simulations for {horizon_years} years (Daily Steps)...")

    Z_s = np.random.normal(0, 1, size=(n_steps, n_simulations))
    Z_v = np.random.normal(0, 1, size=(n_steps, n_simulations))
    dW_v = np.sqrt(dt) * Z_v
    dW_s = np.sqrt(dt) * (rho * Z_v + np.sqrt(1 - rho**2) * Z_s)
    poisson_arrivals = np.random.poisson(lambda_j * dt, size=(n_steps, n_simulations)) if use_jumps else np.zeros((n_steps, n_simulations))
    
    for i in range(n_steps):
        S_t, v_t = portfolio_values[i], variance_paths[i]
        v_t_positive = np.maximum(v_t, 0)
        
        if use_sv:
            d_v = kappa * (theta - v_t_positive) * dt + sigma_v * np.sqrt(v_t_positive) * dW_v[i]
            variance_paths[i+1] = v_t + d_v
            current_variance = v_t_positive
        else:
            variance_paths[i+1] = v0
            current_variance = v0
            
        jump_component = 0
        drift_adjustment = 0
        if use_jumps and np.any(poisson_arrivals[i] > 0):
            has_jump = poisson_arrivals[i] > 0
            jump_size = np.exp(np.random.normal(mu_j, sigma_j, size=n_simulations)) * has_jump
            drift_adjustment = lambda_j * k * dt
            jump_component = (jump_size - 1) * has_jump
        
        drift = (mu - 0.5 * current_variance - drift_adjustment) * dt
        diffusion = np.sqrt(current_variance) * dW_s[i]
        portfolio_values[i+1] = S_t * np.exp(drift + diffusion) * (1 + jump_component)

    print("Heston-Merton simulation complete.")
    monthly_indices = np.linspace(0, n_steps, horizon_years * 12 + 1, dtype=int)
    return portfolio_values[monthly_indices, :], variance_paths[monthly_indices, :]
