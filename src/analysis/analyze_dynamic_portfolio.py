import pandas as pd
from datetime import datetime
import logging
from portfolio_manager import Portfolio

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create portfolio instance
    portfolio = Portfolio(data_directory="data")
    
    # Load transactions
    portfolio.load_transactions("nifty500_dynamic_txns.csv")
    
    # Build holdings and calculate values
    portfolio.build_holdings()
    portfolio.calculate_portfolio_value()
    
    # Get portfolio summary as of November 20, 2024
    summary = portfolio.get_portfolio_summary(as_of_date=datetime(2024, 11, 20).date())
    
    print("\n=== Portfolio Holdings as of Nov 20, 2024 ===")
    print(f"Total Value: ₹{summary['total_value']:,.2f}")
    print(f"Number of Active Positions: {summary['num_positions']}")
    
    print("\nDetailed Holdings:")
    for pos in sorted(summary['positions'], key=lambda x: x['value'], reverse=True):
        print(f"{pos['symbol']}: {pos['quantity']} shares @ ₹{pos['price']:.2f} = ₹{pos['value']:,.2f} ({pos['weight']*100:.1f}%)")
    
    # Print detailed statistics
    print("\n=== Portfolio Statistics ===")
    portfolio.print_statistics()

if __name__ == "__main__":
    main()
