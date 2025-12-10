# main.py
# This script orchestrates the entire quantitative investing workflow.

from src.data_collection import run_data_collection
from src.fundamental_screening import run_fundamental_screening
from src.quant_factor_analysis import run_quant_factor_analysis
from src.correlation_analysis import run_correlation_analysis
from src.portfolio_optimization import run_portfolio_optimization
from src.projection import run_projection
from src.backtesting import run_backtest
from src.monitoring import run_monitoring
from src.portfolio_beta import calculate_portfolio_beta
from src.portfolio_alpha import calculate_portfolio_alpha

def run_investment_pipeline():
    """
    Executes the core investment strategy pipeline (Phases 1-6)
    to generate the final optimized portfolio.
    """
    print("--- Starting Investment Pipeline ---")
    run_data_collection()
    run_fundamental_screening()
    run_quant_factor_analysis()
    run_correlation_analysis()
    run_portfolio_optimization()
    run_projection()
    print("--- Investment Pipeline Finished ---")

def main():
    """
    Main entry point for the application.
    By default, it runs the core investment pipeline.
    Analytical functions (backtest, monitoring) can be run by uncommenting them.
    """
    # This is the main function to run for a real-world rebalance.
    # It generates the latest optimized portfolio weights.
    run_investment_pipeline()

    # --- Analytical Functions (run on demand) ---

    # To run a historical backtest of the entire strategy:
    print("\n--- Starting Analysis: Backtesting ---")
    run_backtest()
    
    # To generate a monitoring report on the latest portfolio's recent performance:
    print("\n--- Starting Analysis: Monitoring Report ---")
    run_monitoring()
    
    # To calculate the portfolio beta:
    print("\n--- Starting Analysis: Portfolio Beta ---")
    calculate_portfolio_beta()
    
    # To calculate the portfolio alpha:
    print("\n--- Starting Analysis: Portfolio Alpha ---")
    calculate_portfolio_alpha()
    run_correlation_analysis()
    print("\nWorkflow finished.")

if __name__ == "__main__":
    main()
