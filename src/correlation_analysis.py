# src/correlation_analysis.py
# This module contains functions for Phase 4: Correlation & Diversification Analysis.

import pandas as pd
import numpy as np
import yaml
import os
from glob import glob
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import timedelta
import networkx as nx
import math
from matplotlib.lines import Line2D

# --- Data Loading ---

def load_scores(data_dir, fundamental_filename, quant_filename):
    """Loads fundamental and quant scores."""
    print("Loading scores from previous phases...")
    fund_scores = pd.read_csv(os.path.join(data_dir, fundamental_filename), index_col=0)
    quant_scores = pd.read_csv(os.path.join(data_dir, quant_filename), index_col='Ticker')
    return fund_scores, quant_scores

def load_price_data(data_dir, prices_filename):
    """Loads the price data CSV."""
    print("Loading price data...")
    prices_path = os.path.join(data_dir, prices_filename)
    return pd.read_csv(prices_path, index_col='Date', parse_dates=True)

def load_latest_fundamental_data(data_dir):
    """Loads the most recent fundamentals .pkl file to get sector info."""
    print("Loading latest fundamental data for sector info...")
    fundamental_files = glob(os.path.join(data_dir, 'fundamentals_*.pkl'))
    if not fundamental_files:
        raise FileNotFoundError("No fundamental data files found.")
    latest_file = max(fundamental_files, key=os.path.getctime)
    return pd.read_pickle(latest_file)

def load_always_include_list(filename):
    """Loads the always-include tickers list."""
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines() if line.strip() and not line.startswith('#')]

# --- Core Logic ---

def calculate_hybrid_score(fund_scores, quant_scores, weights):
    """Calculates the hybrid score from fundamental and quant scores."""
    print("Calculating hybrid scores...")
    aligned_tickers = fund_scores.index.intersection(quant_scores.index)
    
    hybrid_df = pd.DataFrame(index=aligned_tickers)
    hybrid_df['Fundamental_Score'] = fund_scores.loc[aligned_tickers, 'Fundamental_Score']
    hybrid_df['Quant_Score'] = quant_scores.loc[aligned_tickers, 'Quant_Score']
    
    hybrid_df['Hybrid_Score'] = (
        weights['fundamental'] * hybrid_df['Fundamental_Score'] +
        weights['quant'] * hybrid_df['Quant_Score']
    )
    return hybrid_df

def calculate_correlation_matrix(prices_df, tickers, lookback_years):
    """Calculates the return correlation matrix for a list of tickers."""
    print("Calculating correlation matrix...")
    end_date = prices_df.index.max()
    start_date = end_date - timedelta(days=lookback_years * 365)
    
    prices_filtered = prices_df.loc[start_date:end_date, tickers]
    daily_returns = prices_filtered.pct_change().dropna()
    
    return daily_returns.corr()

def filter_correlated_stocks(correlation_matrix, hybrid_scores, fundamentals_data, threshold, keep_percentile, etf_list):
    """Filters out highly correlated stocks using a graph-based clustering approach."""
    print("Filtering highly correlated stocks by sector...")

    sectors = {t: fundamentals_data[t]['info'].get('sector', 'Unknown') for t in hybrid_scores.index if t in fundamentals_data}
    hybrid_scores['Sector'] = hybrid_scores.index.map(sectors)
    hybrid_scores['Sector'] = hybrid_scores['Sector'].fillna('Unknown')


    tickers_to_drop = set()
    all_clusters = []

    # Process each sector independently
    for sector, group in hybrid_scores.groupby('Sector'):
        if len(group) < 2:
            continue

        print(f"\nProcessing sector: {sector} ({len(group)} stocks)")

        # Create a graph for the current sector
        G = nx.Graph()
        sector_tickers = group.index.tolist()
        G.add_nodes_from(sector_tickers)

        # Add edges for correlations above the threshold
        corr_subset = correlation_matrix.loc[sector_tickers, sector_tickers]
        for i in range(len(sector_tickers)):
            for j in range(i + 1, len(sector_tickers)):
                ticker1 = sector_tickers[i]
                ticker2 = sector_tickers[j]
                if corr_subset.loc[ticker1, ticker2] > threshold:
                    G.add_edge(ticker1, ticker2)

        # Find connected components (correlated clusters)
        clusters = list(nx.connected_components(G))
        all_clusters.extend(clusters)


        for cluster in clusters:
            if len(cluster) > 1:
                print(f"  - Found correlated cluster of {len(cluster)} stocks: {list(cluster)}")

                # Sort stocks in the cluster by Hybrid Score
                cluster_scores = hybrid_scores.loc[list(cluster)]
                sorted_cluster = cluster_scores.sort_values(by='Hybrid_Score', ascending=False)

                # Determine how many stocks to keep
                n_keep = max(1, math.ceil(len(cluster) * keep_percentile))

                # Identify stocks to drop
                stocks_to_drop_from_cluster = sorted_cluster.index[n_keep:]
                tickers_to_drop.update(stocks_to_drop_from_cluster)

                print(f"    - Keeping top {n_keep} stock(s): {sorted_cluster.index[:n_keep].tolist()}")
                if len(stocks_to_drop_from_cluster) > 0:
                    print(f"    - Dropping {len(stocks_to_drop_from_cluster)} stock(s): {stocks_to_drop_from_cluster.tolist()}")

    diversified_tickers = list(set(hybrid_scores.index) - tickers_to_drop)

    # Ensure at least one ETF is included
    etfs_in_diversified = [t for t in diversified_tickers if t in etf_list]
    if not etfs_in_diversified:
        print("\n  - No ETFs in diversified list. Adding best-scoring ETF...")
        all_etfs_with_scores = hybrid_scores[hybrid_scores.index.isin(etf_list)].sort_values(by='Hybrid_Score', ascending=False)
        if not all_etfs_with_scores.empty:
            best_etf = all_etfs_with_scores.index[0]
            diversified_tickers.append(best_etf)
            print(f"    - Added ETF: {best_etf} (Score: {all_etfs_with_scores.loc[best_etf, 'Hybrid_Score']:.4f})")
        else:
            print("    - Warning: No ETFs available to add!")

    print(f"\nCorrelation filtering complete. {len(diversified_tickers)} tickers remain (before adding always-include).")
    return diversified_tickers, hybrid_scores, tickers_to_drop, all_clusters


# src/correlation_analysis.py
# Replace the add_always_include_stocks function:

def add_always_include_stocks(diversified_tickers, always_include_list, prices_df, fundamentals_data, min_price_usd, etf_list):
    """
    Adds always-include stocks to the diversified list, ensuring they pass price filter.
    This happens AFTER all other filtering.
    """
    print("\n--- Adding Always-Include Stocks ---")
    
    # Filter always-include stocks by price
    valid_always_include = []
    for ticker in always_include_list:
        if ticker not in prices_df.columns:
            print(f"  - Warning: Always-include ticker {ticker} not in price data")
            continue
            
        # Get the latest available price
        latest_price = prices_df[ticker].dropna().iloc[-1] if not prices_df[ticker].dropna().empty else 0
        
        # Check if it's an ETF (ETFs don't need fundamental data)
        is_etf = ticker in etf_list
        
        # For stocks, check if we have fundamental data for currency conversion
        if not is_etf:
            if ticker in fundamentals_data and 'info' in fundamentals_data[ticker]:
                currency = fundamentals_data[ticker]['info'].get('currency', 'USD').upper()
                if currency != 'USD':
                    # Convert to USD
                    try:
                        from currency_converter import CurrencyConverter
                        c = CurrencyConverter(fallback_on_missing_rate=True)
                        latest_price = c.convert(latest_price, currency, 'USD')
                    except Exception as e:
                        print(f"  - Warning: Could not convert {ticker} from {currency}: {e}")
        
        if latest_price >= min_price_usd:
            valid_always_include.append(ticker)
            print(f"  - Always-include {ticker} passed price filter: ${latest_price:.2f}")
        else:
            print(f"  - Always-include {ticker} failed price filter: ${latest_price:.2f} (min: ${min_price_usd})")
    
    if not valid_always_include:
        print("  - No always-include stocks passed the price filter.")
        return diversified_tickers
    
    # Add them to the final list
    added_count = 0
    for ticker in valid_always_include:
        if ticker not in diversified_tickers:
            diversified_tickers.append(ticker)
            added_count += 1
            print(f"  - Added always-include ticker: {ticker}")
        else:
            print(f"  - Always-include ticker {ticker} already in diversified list")
    
    print(f"\n  - Total always-include stocks added: {added_count}")
    print(f"  - Final ticker count: {len(diversified_tickers)}")
    
    return diversified_tickers    
# --- Output Generation ---

def visualize_correlation_graph(hybrid_scores, correlation_matrix, min_threshold, tickers_to_drop, clusters, output_path):
    """
    Creates and saves a visualization of the correlation graph.
    - Nodes are colored by sector.
    - Node size is proportional to Hybrid Score.
    - Dropped nodes are semi-transparent.
    - Edges represent correlation > min_threshold, with style indicating strength.
    - Clustered nodes have a highlighted border.
    """
    print("Generating enhanced correlation graph visualization...")

    all_tickers = hybrid_scores.index.tolist()
    G = nx.Graph()
    G.add_nodes_from(all_tickers)

    # --- Edge Attributes ---
    pos_edges, neg_edges = [], []
    pos_weights, neg_weights = [], []
    pos_alphas, neg_alphas = [], []
    
    for i in range(len(all_tickers)):
        for j in range(i + 1, len(all_tickers)):
            ticker1, ticker2 = all_tickers[i], all_tickers[j]
            if ticker1 in correlation_matrix.index and ticker2 in correlation_matrix.index:
                corr = correlation_matrix.loc[ticker1, ticker2]
                if abs(corr) > min_threshold:
                    # Scale width and alpha based on absolute correlation
                    # Using a non-linear scale (power of 2) to make stronger correlations stand out more
                    width = 0.2 + 4 * (abs(corr) ** 2)
                    alpha = 0.1 + 0.8 * (abs(corr) ** 2)
                    
                    if corr > 0:
                        pos_edges.append((ticker1, ticker2))
                        pos_weights.append(width)
                        pos_alphas.append(alpha)
                    else:
                        neg_edges.append((ticker1, ticker2))
                        neg_weights.append(width)
                        neg_alphas.append(alpha)

    # --- Node Attributes ---
    # 1. Sector Colors
    unique_sectors = sorted(hybrid_scores['Sector'].dropna().unique())
    colors = plt.cm.get_cmap('tab20', len(unique_sectors))
    sector_color_map = {sector: colors(i) for i, sector in enumerate(unique_sectors)}
    node_colors = [sector_color_map.get(hybrid_scores.loc[t, 'Sector'], 'grey') for t in G.nodes()]

    # 2. Node Sizes (proportional to Hybrid Score)
    min_size, max_size = 150, 2500
    scores = hybrid_scores.loc[list(G.nodes()), 'Hybrid_Score'].fillna(0.5)
    min_score, max_score = scores.min(), scores.max()
    if min_score == max_score: # Avoid division by zero if all scores are the same
        node_sizes = [min_size] * len(G.nodes())
    else:
        node_sizes = [min_size + (s - min_score) / (max_score - min_score) * (max_size - min_size) for s in scores]

    # 3. Border color and width for clusters
    node_borders = []
    border_widths = []
    for node in G.nodes():
        is_in_cluster = any(node in cluster for cluster in clusters if len(cluster) > 1)
        if is_in_cluster:
            node_borders.append('gold')
            border_widths.append(3)
        else:
            node_borders.append('black')
            border_widths.append(1)

    # --- Drawing ---
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(32, 32))
    pos = nx.spring_layout(G, k=0.6, iterations=70, seed=42)

    # Refactor node drawing for efficiency
    node_to_props = {node: {'color': node_colors[i], 'size': node_sizes[i], 'border': node_borders[i], 'width': border_widths[i]} for i, node in enumerate(G.nodes())}
    
    kept_nodes = [n for n in G.nodes() if n not in tickers_to_drop]
    dropped_nodes = [n for n in G.nodes() if n in tickers_to_drop]
    
    # Draw kept nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=kept_nodes,
                           node_color=[node_to_props[n]['color'] for n in kept_nodes],
                           node_size=[node_to_props[n]['size'] for n in kept_nodes],
                           edgecolors=[node_to_props[n]['border'] for n in kept_nodes],
                           linewidths=[node_to_props[n]['width'] for n in kept_nodes],
                           alpha=1.0)
                           
    # Draw dropped nodes with transparency
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=dropped_nodes,
                           node_color=[node_to_props[n]['color'] for n in dropped_nodes],
                           node_size=[node_to_props[n]['size'] for n in dropped_nodes],
                           edgecolors=[node_to_props[n]['border'] for n in dropped_nodes],
                           linewidths=[node_to_props[n]['width'] for n in dropped_nodes],
                           alpha=0.35)

    # Draw edges with dynamic styles
    for i, edge in enumerate(pos_edges):
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=[edge], width=pos_weights[i], alpha=pos_alphas[i], edge_color='skyblue')
        
    for i, edge in enumerate(neg_edges):
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=[edge], width=neg_weights[i], alpha=neg_alphas[i], edge_color='tomato')

    # Draw labels
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_weight='bold', font_color='white')
    
    ax.set_title('Correlation Graph of Stock Universe', fontsize=40, color='white')
    
    # --- Legend ---
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=sector,
                              markerfacecolor=color, markersize=15) for sector, color in sector_color_map.items()]
    legend_elements.append(Line2D([0], [0], color='skyblue', lw=4, label='Positive Correlation'))
    legend_elements.append(Line2D([0], [0], color='tomato', lw=4, label='Negative Correlation'))
    legend_elements.append(Line2D([0], [0], marker='o', color='none', label='In Cluster',
                                  markeredgecolor='gold', markeredgewidth=3, markersize=15))
    legend_elements.append(Line2D([0], [0], marker='o', color='none', label='Dropped',
                                  markerfacecolor='grey', alpha=0.35, markersize=15))
    
    legend = ax.legend(handles=legend_elements, title="Legend", loc='upper right', fontsize=16, title_fontsize=18)
    plt.setp(legend.get_texts(), color='white')
    plt.setp(legend.get_title(), color='white')

    fig.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    plt.style.use('default') # Reset style
    print(f"Correlation graph saved to {output_path}")


# --- Main Phase 4 Orchestrator ---

# src/correlation_analysis.py
# Update the save_outputs function and run_correlation_analysis function

def save_outputs(hybrid_scores, diversified_tickers, correlation_matrix, data_dir, reports_dir, config, prices=None, lookback_years=None):
    """Saves all outputs for Phase 4."""
    print("Saving all outputs for Phase 4...")
    
    # Save hybrid scores
    hybrid_scores.sort_values(by='Hybrid_Score', ascending=False).to_csv(os.path.join(data_dir, config['output_hybrid_scores_filename']))
    print(f"Hybrid scores saved to {os.path.join(data_dir, config['output_hybrid_scores_filename'])}")
    
    # Save diversified tickers
    with open(os.path.join(data_dir, config['output_tickers_filename']), 'w') as f:
        for ticker in sorted(diversified_tickers):
            f.write(f"{ticker}\n")
    print(f"Diversified tickers saved to {os.path.join(data_dir, config['output_tickers_filename'])}")
    
    # Save correlation heatmap - recalculate correlation matrix for final tickers if needed
    try:
        # Try to create heatmap with the current correlation matrix
        available_tickers = [t for t in diversified_tickers if t in correlation_matrix.index]
        if len(available_tickers) < len(diversified_tickers):
            print(f"  - Warning: Correlation matrix missing {len(diversified_tickers) - len(available_tickers)} tickers. Recalculating...")
            if prices is not None and lookback_years is not None:
                correlation_matrix = calculate_correlation_matrix(prices, diversified_tickers, lookback_years)
            else:
                print("  - Cannot recalculate correlation matrix, skipping heatmap")
                return
        
        plt.figure(figsize=(20, 20))
        sns.heatmap(correlation_matrix.loc[diversified_tickers, diversified_tickers], cmap='coolwarm', annot=False)
        plt.title('Correlation Matrix of Diversified Stocks')
        plt.savefig(os.path.join(reports_dir, config['output_heatmap_filename']))
        plt.close()
        print(f"Correlation heatmap saved to {os.path.join(reports_dir, config['output_heatmap_filename'])}")
    except KeyError as e:
        print(f"  - Warning: Could not create correlation heatmap: {e}")
        print("  - Skipping heatmap generation")
    
    # Save sector distribution
    sector_dist = hybrid_scores.loc[[t for t in diversified_tickers if t in hybrid_scores.index], 'Sector'].value_counts(normalize=True).reset_index()
    sector_dist.columns = ['Sector', 'Weight']
    sector_dist.to_csv(os.path.join(data_dir, config['output_sector_dist_filename']), index=False)
    print(f"Sector distribution saved to {os.path.join(data_dir, config['output_sector_dist_filename'])}")

# --- Main Phase 4 Orchestrator ---

def run_correlation_analysis(config_path="config.yaml"):
    """Main function to run the entire correlation and diversification phase."""
    print("\n--- Starting Phase 4: Correlation & Diversification Analysis ---")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        data_config, screening_config, quant_config, corr_config = \
            config['data_collection'], config['fundamental_screening'], config['quant_factor_analysis'], config['correlation_analysis']

    fund_scores, quant_scores = load_scores(data_config['output_dir'], screening_config['output_scores_filename'], quant_config['output_filename'])
    prices = load_price_data(data_config['output_dir'], data_config['prices_filename'])
    fundamentals = load_latest_fundamental_data(data_config['output_dir'])
    
    # Load ETF list
    with open(data_config['etf_list_filename'], 'r') as f:
        etf_list = [line.strip() for line in f.readlines() if line.strip()]
    
    hybrid_scores = calculate_hybrid_score(fund_scores, quant_scores, corr_config['hybrid_score_weights'])
    # Initial correlation matrix for all scored stocks
    initial_correlation_matrix = calculate_correlation_matrix(prices, hybrid_scores.index.tolist(), corr_config['lookback_years'])
    
    diversified_tickers, hybrid_scores_with_sector, tickers_to_drop, all_clusters = filter_correlated_stocks(
        initial_correlation_matrix, hybrid_scores, fundamentals,
        corr_config['correlation_threshold'],
        corr_config['correlation_keep_percentile'],
        etf_list
    )
    
    # Visualize the graph BEFORE adding the 'always-include' stocks for a clearer picture of the filtering logic
    visualize_correlation_graph(
        hybrid_scores_with_sector,
        initial_correlation_matrix,
        corr_config['graph_min_corr_threshold'],
        tickers_to_drop,
        all_clusters,
        os.path.join(data_config['reports_dir'], corr_config['output_graph_filename'])
    )
    
    # NOW add always-include stocks AFTER all filtering
    always_include_list = load_always_include_list(screening_config['always_include_filename'])
    final_diversified_tickers = add_always_include_stocks(
        diversified_tickers,
        always_include_list,
        prices,
        fundamentals,
        data_config['min_stock_price'],
        etf_list
    )
    
    # Update hybrid_scores_with_sector to include the always-include stocks that were added
    for ticker in final_diversified_tickers:
        if ticker not in hybrid_scores_with_sector.index:
            sector = 'Unknown'
            if ticker in fundamentals and 'info' in fundamentals[ticker]:
                sector = fundamentals[ticker]['info'].get('sector', 'Unknown')
            elif ticker in etf_list:
                sector = 'ETF'
            
            hybrid_scores_with_sector.loc[ticker] = {
                'Fundamental_Score': 0.5, 'Quant_Score': 0.5, 'Hybrid_Score': 0.5, 'Sector': sector
            }
    
    # Recalculate correlation matrix for the final set of tickers for the heatmap
    final_correlation_matrix = calculate_correlation_matrix(prices, final_diversified_tickers, corr_config['lookback_years'])
    
    save_outputs(hybrid_scores_with_sector, final_diversified_tickers, final_correlation_matrix, data_config['output_dir'], data_config['reports_dir'], corr_config)
    
    print("--- Phase 4 Finished ---")