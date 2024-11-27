import pandas as pd
import logging
from portfolio_manager import Portfolio
from datetime import datetime

def print_portfolio_summary(name: str, portfolio: Portfolio):
    """Print a summary of the portfolio"""
    summary = portfolio.get_portfolio_summary()
    
    print(f"\n=== {name} Summary ===")
    print(f"Total Value: ₹{summary['total_value']:,.2f}")
    print(f"Number of Positions: {summary['num_positions']}")
    
    print("\nTop 5 Positions:")
    for pos in summary['positions'][:5]:
        print(f"{pos['symbol']}: {pos['quantity']} shares @ ₹{pos['price']:.2f} = ₹{pos['value']:,.2f} ({pos['weight']*100:.1f}%)")
    
    # Print detailed statistics
    print(f"\n=== {name} Statistics ===")
    portfolio.print_statistics()

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create two portfolio instances
    portfolio1 = Portfolio(data_directory="data")
    portfolio2 = Portfolio(data_directory="data")
    
    # Load different transaction files for each portfolio
    portfolio1.load_transactions("sample_transactions.csv")
    portfolio2.load_transactions("portfolio2_transactions.csv")
    
    # Build holdings and calculate values for both portfolios
    for portfolio in [portfolio1, portfolio2]:
        portfolio.build_holdings()
        portfolio.calculate_portfolio_value()
        if portfolio.portfolio_value is None:
            logging.error(f"Failed to calculate portfolio value. Check if price data exists for all symbols.")
            return
    
    # Print summaries with statistics
    print_portfolio_summary("Portfolio 1 (Original)", portfolio1)
    print_portfolio_summary("Portfolio 2 (Tech Focus)", portfolio2)
    
    # Compare portfolio values over time
    print("\n=== Portfolio Value Comparison ===")
    comparison = pd.DataFrame({
        'Portfolio 1': portfolio1.portfolio_value['total_value'],
        'Portfolio 2': portfolio2.portfolio_value['total_value']
    })
    
    # Find the latest date available in both portfolios
    latest_date = min(portfolio1.portfolio_value.index[-1], portfolio2.portfolio_value.index[-1])
    earliest_date = max(portfolio1.portfolio_value.index[0], portfolio2.portfolio_value.index[0])
    
    print(f"\nComparison Period: {earliest_date} to {latest_date}")
    print(f"Portfolio 1 Value: ₹{comparison['Portfolio 1'].loc[latest_date]:,.2f}")
    print(f"Portfolio 2 Value: ₹{comparison['Portfolio 2'].loc[latest_date]:,.2f}")
    
    # Calculate returns using the common date range
    p1_return = (comparison['Portfolio 1'].loc[latest_date] / comparison['Portfolio 1'].loc[earliest_date] - 1) * 100
    p2_return = (comparison['Portfolio 2'].loc[latest_date] / comparison['Portfolio 2'].loc[earliest_date] - 1) * 100
    
    print(f"\nReturns ({earliest_date} to {latest_date}):")
    print(f"Portfolio 1: {p1_return:.1f}%")
    print(f"Portfolio 2: {p2_return:.1f}%")

if __name__ == "__main__":
    main()
