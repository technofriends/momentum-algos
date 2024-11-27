import pandas as pd
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class PortfolioBuilder:
    def __init__(self, transaction_file: str, start_date: str = "1990-01-01"):
        """
        Initialize the PortfolioBuilder with a transaction file and start date
        
        Args:
            transaction_file (str): Path to the transaction CSV file
            start_date (str): Start date for the portfolio in YYYY-MM-DD format
        """
        self.transaction_file = transaction_file
        self.start_date = pd.to_datetime(start_date)
        self.transactions = None
        self.holdings = None
    
    def read_transactions(self):
        """Read and validate the transaction file"""
        try:
            # Read transactions
            self.transactions = pd.read_csv(self.transaction_file)
            
            # Validate required columns
            required_columns = ['date', 'qty', 'symbol', 'price']
            missing_columns = [col for col in required_columns if col not in self.transactions.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Convert date to datetime
            self.transactions['date'] = pd.to_datetime(self.transactions['date'])
            
            # Sort by date
            self.transactions = self.transactions.sort_values('date')
            
            # Validate data types
            self.transactions['qty'] = pd.to_numeric(self.transactions['qty'])
            self.transactions['price'] = pd.to_numeric(self.transactions['price'])
            
            logger.info(f"Successfully read {len(self.transactions)} transactions")
            return True
            
        except Exception as e:
            logger.error(f"Error reading transaction file: {str(e)}")
            raise
    
    def build_holdings(self):
        """Build a holdings dataframe with daily positions"""
        if self.transactions is None:
            self.read_transactions()
        
        try:
            # Get unique symbols and date range
            symbols = self.transactions['symbol'].unique()
            date_range = pd.date_range(
                start=min(self.start_date, self.transactions['date'].min()),
                end=self.transactions['date'].max(),
                freq='D'
            )
            
            # Initialize holdings dataframe with 0s
            self.holdings = pd.DataFrame(0, index=date_range, columns=symbols)
            
            # Process each transaction
            for _, row in self.transactions.iterrows():
                date = row['date']
                symbol = row['symbol']
                qty = row['qty']
                
                # Update holdings from this date forward
                self.holdings.loc[date:, symbol] += qty
            
            # Remove dates with no holdings
            self.holdings = self.holdings.loc[
                (self.holdings != 0).any(axis=1)
            ]
            
            logger.info(f"Built holdings dataframe with {len(self.holdings)} dates and {len(symbols)} symbols")
            return self.holdings
            
        except Exception as e:
            logger.error(f"Error building holdings: {str(e)}")
            raise
    
    def get_holdings_for_date(self, date: str) -> pd.Series:
        """Get holdings snapshot for a specific date"""
        if self.holdings is None:
            self.build_holdings()
        
        date = pd.to_datetime(date)
        if date < self.holdings.index.min():
            return pd.Series(0, index=self.holdings.columns)
        
        # Get the last holdings up to the given date
        holdings = self.holdings.loc[:date].iloc[-1]
        return holdings[holdings != 0]  # Return only non-zero positions
    
    def save_holdings(self, output_file: str):
        """Save holdings to a CSV file"""
        if self.holdings is None:
            self.build_holdings()
        
        # Reset index to make date a column, then save
        holdings_df = self.holdings.copy()
        holdings_df.index.name = 'date'  # Name the index
        holdings_df.to_csv(output_file)
        logger.info(f"Saved holdings to {output_file}")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Use sample transactions file
    builder = PortfolioBuilder("sample_transactions.csv")
    holdings = builder.build_holdings()
    
    # Print some sample outputs
    print("\nFull Holdings DataFrame:")
    print(holdings)
    
    print("\nHoldings on 2023-03-01:")
    print(builder.get_holdings_for_date("2023-03-01"))
    
    print("\nHoldings on 2023-05-15:")
    print(builder.get_holdings_for_date("2023-05-15"))
    
    # Save to CSV
    builder.save_holdings("data/processed/holdings.csv")
