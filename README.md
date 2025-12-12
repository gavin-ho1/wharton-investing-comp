# Wharton Investing Competition Codebase

## This repository contains all trading strategies, stock filtering, portfolio allocations, backesting, and projections used in the 2025-2026 [Wharton Global High School Investment Competition](https://globalyouth.wharton.upenn.edu/competitions/investment-competition/) along with detailed descriptions of each part of the code. This repository will go public after the final report submission date.

It contains a multi-phase quantitative investing workflow designed to identify promising stocks, construct an optimized portfolio, and project its future performance. The entire process is configurable via the `config.yaml` file. It also contains the code we used to generate various graphs and tables used in our final report.

## Investment Pipeline

The investment pipeline consists of the following phases:

1.  **Data Collection:** Gathers historical price data, dividends, and fundamental financial data for a universe of stocks.
2.  **Fundamental Screening:** Ranks stocks based on fundamental metrics like value, growth, and quality.
3.  **Quantitative Factor Analysis:** Scores stocks based on quantitative factors such as momentum, volatility, and Sharpe ratio.
4.  **Correlation & Diversification Analysis:** Analyzes the correlation between stocks and ensures diversification across sectors.
5.  **Portfolio Optimization:** Constructs an optimal portfolio based on a specified objective function (e.g., maximizing Sharpe ratio).
6.  **Projection:** Runs a Monte Carlo simulation to project the future performance of the optimized portfolio.
7.  **Backtesting:** Performs a historical backtest of the investment strategy.
8.  **Monitoring:** Generates a monitoring report on the recent performance of the portfolio.

## Configuration

The behavior of the investment pipeline can be customized by editing the `config.yaml` file. This file contains parameters for each phase of the pipeline, such as lookback periods, scoring weights, and optimization settings.

## Usage

To run the main investment pipeline, execute the following command:

```bash
python3 main.py
```

The `main.py` script can also be used to run the analytical functions (backtesting and monitoring) by uncommenting the relevant lines in the `main()` function.

## Scoring Equations

The scoring methodology is divided into two main components: a **Fundamental Score** and a **Quant Score**. These scores are calculated independently, ranked, and then combined to create a final hybrid score used for stock selection.

---

### Fundamental Score

The Fundamental Score evaluates stocks based on their financial health and business performance. It is derived from three sub-categories: Value, Growth, and Quality.

#### 1. Raw Metric Calculation

Six key financial metrics are calculated for each stock.

*   **P/E (Price-to-Earnings) Ratio:**

$$
\text{P/E Ratio} = \frac{\text{Market Capitalization}}{\text{Net Income}}
$$

*   **P/B (Price-to-Book) Ratio:**

$$
\text{P/B Ratio} = \frac{\text{Market Capitalization}}{\text{Total Stockholder Equity}}
$$

*   **EV/EBITDA Ratio:**

$$
\text{EV/EBITDA} = \frac{\text{Market Cap} + \text{Total Debt} - \text{Cash}}{\text{EBIT} + \text{Depreciation}}
$$

*   **Revenue CAGR (5-Year):**

$$
\text{Revenue CAGR} = \left( \frac{\text{Ending Revenue}}{\text{Starting Revenue}} \right)^{\frac{1}{5}} - 1
$$

*   **EPS CAGR (5-Year):**

$$
\text{EPS CAGR} = \left( \frac{\text{Ending EPS}}{\text{Starting EPS}} \right)^{\frac{1}{5}} - 1
$$

*   **Return on Equity (ROE):**

$$
\text{ROE} = \frac{\text{Net Income}}{\text{Total Stockholder Equity}}
$$

#### 2. Percentile Ranking

Each raw metric is converted into a percentile rank, $R$, from 0.0 to 1.0. For metrics where a lower value is better (P/E, P/B, EV/EBITDA), the rank is inverted. Missing values receive a neutral rank of 0.5.

$$
R_{\text{inverted}} = 1.0 - R_{\text{original}}
$$

#### 3. Composite Score Calculation

The percentile ranks ($R$) are first averaged within their respective categories:

$$
S_{\text{Value}} = \text{mean}(R_{P/E}, R_{P/B}, R_{EV/EBITDA})
$$
$$
S_{\text{Growth}} = \text{mean}(R_{\text{Revenue CAGR}}, R_{\text{EPS CAGR}})
$$
$$
S_{\text{Quality}} = R_{ROE}
$$

The final Fundamental Score is a weighted average of these category scores, with weights ($w$) defined in `config.yaml`:

$$
\text{Fundamental Score} = (w_{\text{value}} \cdot S_{\text{Value}}) + (w_{\text{growth}} \cdot S_{\text{Growth}}) + (w_{\text{quality}} \cdot S_{\text{Quality}})
$$

---

### Quant Score

The Quant Score evaluates stocks based on market-driven (price-based) factors.

#### 1. Raw Factor Calculation

Four quantitative factors are calculated from historical price data.

*   **Momentum (12-1):**

$$
\text{Momentum} = \frac{\text{Price}_{t-1}}{\text{Price}_{t-13}} - 1
$$

*   **Volatility (Annualized):**

$$
\text{Volatility} = \sigma_{\text{daily returns}} \cdot \sqrt{252}
$$

*   **Beta:**

$$
\beta = \frac{\text{Cov}(R_{\text{stock}}, R_{\text{benchmark}})}{\text{Var}(R_{\text{benchmark}})}
$$

*   **Sharpe Ratio (Annualized):**

$$
\text{Sharpe Ratio} = \frac{\text{Annualized Return} - R_f}{\text{Annualized Volatility}}
$$

#### 2. Factor Ranking

Similar to the fundamental metrics, each factor is converted to a percentile rank ($R$). For Volatility and Beta, the rank is inverted.

#### 3. Composite Score Calculation

The final Quant Score is a weighted average of the factor ranks, with weights ($w$) defined in `config.yaml`:

$$
\text{Quant Score} = (w_{\text{mom}} \cdot R_{\text{Mom}}) + (w_{\text{vol}} \cdot R_{\text{Vol}}) + (w_{\text{beta}} \cdot R_{\text{Beta}}) + (w_{\text{sharpe}} \cdot R_{\text{Sharpe}})
$$

---

### Composite Score

The Composite Score is the final, unified metric used to rank stocks before the diversification phase. It provides a holistic measure of a stock's attractiveness by combining its fundamental strength with its market-based (quantitative) characteristics.

It is calculated as a weighted average of the Fundamental Score and the Quant Score, with weights ($w$) defined in `config.yaml`:

$$
\text{Composite Score} = (w_{\text{fundamental}} \cdot \text{Fundamental Score}) + (w_{\text{quant}} \cdot \text{Quant Score})
$$

---

## Correlation & Diversification

After scoring, the script implements a sophisticated filtering process to enhance portfolio diversification. This phase aims to reduce concentration risk by removing stocks that are highly correlated with their peers within the same sector. The process uses a graph-based clustering approach.

### 1. Ranking by Composite Score

The filtering process relies on the **Composite Score** (calculated in the previous section) to provide a single, holistic measure of each stock's attractiveness. Stocks with higher Composite Scores are preferentially kept.

### 2. Intra-Sector Clustering

The core of the diversification logic operates independently on each economic sector. This ensures that the filtering process compares apples to apples (e.g., tech stocks with other tech stocks).

For each sector, the process is as follows:

-   **Build a Correlation Graph:** A graph is constructed where each stock in the sector is a node. An edge is created between two nodes if their historical return correlation exceeds a specified `correlation_threshold` from `config.yaml`.

-   **Identify Correlated Clusters:** The algorithm then identifies all "connected components" in the graph. A connected component is a subgraph where every node is reachable from every other node. These components represent clusters of highly correlated stocks.

-   **Filter Within Clusters:** For each identified cluster, the stocks are ranked by their Composite Score. The script then keeps only the top-performing stocks, determined by the `correlation_keep_percentile` parameter. For example, if a cluster has 10 stocks and the percentile is 0.2, only the top 2 stocks (ranked by Composite Score) are retained for the next phase. The rest are filtered out. A minimum of one stock is always kept from any cluster.

This methodology ensures that the final stock universe is not only composed of high-scoring stocks but is also diversified within each sector, reducing the risk of holding multiple similar assets that are likely to perform in the same way.

---

## Portfolio Optimization

### Mean-Variance Optimization

The script's primary portfolio optimization strategy is Mean-Variance Optimization, a cornerstone of Modern Portfolio Theory. The goal is to find the portfolio asset allocation that maximizes the investor's utility by balancing expected return against risk (variance).

The optimization is performed by the `pypfopt` library, which solves for the weights (`w`) that maximize the following quadratic utility function:

$$
\max_{w} \quad w^T \mu - \frac{\delta}{2} w^T \Sigma w
$$

Where:
- **$w$** : The vector of portfolio weights (the variables to be solved for).
- **$w^T$** : The transpose of the weight vector.
- **$\mu$** : The vector of expected returns for each asset.
- **$\Sigma$** : The covariance matrix of asset returns, which models the risk of the assets and their co-movements.
- **$\delta$** : The risk aversion parameter, which is set in `config.yaml`. This value quantifies the investor's penalty for taking on risk. A higher $\delta$ leads to a more conservative (lower-risk) portfolio.

The optimizer seeks to find the weights $w$ that maximize this function, subject to constraints such as weight bounds (e.g., long-only, max weight per stock).

---

## Monte Carlo Projection

Phase 6 of the pipeline involves a sophisticated Monte Carlo simulation to project the long-term performance of the optimized portfolio. This provides a probabilistic forecast of future outcomes, helping to understand the potential range of returns and risks. The projection is highly configurable and supports several advanced simulation models.

### Core Simulation Logic

The simulation engine projects the portfolio value over a specified time horizon (e.g., 10 years) by running thousands of independent simulations. Each simulation path is generated by modeling asset or portfolio returns using stochastic differential equations.

### Key Configuration Options

-   **`simulation_model`**: Selects the core engine for the simulation.
-   **`n_simulations`**: The number of simulation paths to generate (e.g., 5,000).
-   **`horizon_years`**: The length of the projection period.
-   **`initial_investment`**: The starting dollar value of the portfolio.
-   **`enable_monthly_rebalancing`**: A boolean flag to enable a dynamic rebalancing strategy.

### Simulation Models

The script supports three distinct simulation models, selectable in `config.yaml`:

#### 1. Geometric Brownian Motion (GBM) - Multi-Asset

This is a widely-used model that assumes asset returns follow a random walk. It is simulated on a **monthly** time step. The change in the price of an asset, $S_i$, is modeled as:

$$
dS_i = \mu_i S_i dt + \sigma_i S_i dW_i
$$

Where:
-   **$\mu_i$**: The expected annual return (drift) for asset $i$.
-   **$\sigma_i$**: The annual volatility for asset $i$.
-   **$dt$**: The time step.
-   **$dW_i$**: A Wiener process (random shock) for asset $i$.

The model incorporates the correlation between assets by using a Cholesky decomposition of the historical covariance matrix to generate correlated random shocks ($dW$).

#### 2. Heston-Merton Model - Portfolio-Level

This more advanced model treats the entire portfolio as a single asset and introduces two key features: stochastic volatility and jump diffusion, simulated on a **daily** time step.

-   **Stochastic Volatility (Heston):** Unlike GBM, which assumes constant volatility, the Heston model allows volatility itself to be a random variable. This captures the real-world phenomenon of volatility clustering (periods of high volatility followed by more high volatility). The variance, $v_t$, follows its own stochastic process:

$$
dv_t = \kappa (\theta - v_t) dt + \sigma_v \sqrt{v_t} dW_t^{(v)}
$$

-   **Jump Diffusion (Merton):** This component adds sudden, discontinuous "jumps" to the portfolio return, modeling the impact of unexpected major news or events. The jump size and frequency are controlled by a Poisson process.

#### 3. Heston-Merton Model - Asset-Level

This is the most sophisticated model available. It simulates each individual asset's price path using its own Heston-Merton process while maintaining the correlation structure between them. This approach provides a more granular and realistic simulation of the portfolio's evolution. Key features include:

-   **Systemic and Idiosyncratic Shocks:** The model includes both market-wide (systemic) jumps that affect all assets and asset-specific (idiosyncratic) jumps.
-   **Shared Volatility Process:** A single, market-wide stochastic volatility process is simulated, and each asset's individual volatility is scaled relative to it.
-   **Correlated Returns:** Asset returns remain correlated, driven by a shared, correlated Wiener process.

### Dynamic Monthly Rebalancing

When `enable_monthly_rebalancing` is set to `true`, the simulation adjusts the portfolio weights at the start of each simulated month. This hybrid strategy blends the initial, fundamentally-derived weights with a momentum factor based on the previous month's simulated returns. The rebalancing logic is governed by an `alpha` parameter, which determines the weighting between the fundamental and momentum components.

### Projection Outputs

The primary output of this phase is a multi-plot "tearsheet" visualization (`projection_tearsheet.png`) that provides a comprehensive overview of the simulation results. It includes:
1.  **Simulation Paths Plot:** A log-scaled chart showing a sample of the simulated portfolio value paths over time, with key percentiles (10th, 50th, 90th) highlighted.
2.  **Final Value Distribution:** A histogram showing the distribution of the final portfolio values across all simulations.
3.  **Percentile Statistics Table:** A table summarizing key performance metrics (CAGR, Volatility, Sharpe Ratio) for different percentile outcomes.
4.  **Benchmark Comparison Table:** A table comparing the average performance of the simulation against a historical equivalent period for a benchmark ticker (e.g., SPY).

Additional outputs, including a summary CSV and standalone versions of the plots, are also saved to the `reports/` and `data/` directories. Images used in our final report, along with the code used to generate them can be found in `final-images/`.
