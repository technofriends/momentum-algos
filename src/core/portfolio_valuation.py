import pandas as pd
import numpy as np
import os
from pathlib import Path
import logging

class PortfolioValuation:
    def __init__(self, holdings_file, data_directory):
        """
        Initialize portfolio valuation with holdings and price data
        
        Args:
            holdings_file (str): Path to holdings CSV file
            data_directory (str): Path to directory containing price data files
        """
        # Read holdings and convert index to date
        self.holdings = pd.read_csv(holdings_file, index_col='date', parse_dates=True)
        self.holdings.index = self.holdings.index.date
        self.data_directory = Path(data_directory)
        self.price_data = None
        self.portfolio_value = None
        
    def load_price_data(self):
        """Load and combine price data for all symbols in holdings"""
        # Get unique symbols from holdings
        symbols = self.holdings.columns.tolist()
        
        # Initialize empty dataframe for prices
        self.price_data = pd.DataFrame()
        
        for symbol in symbols:
            price_file = self.data_directory / f"{symbol.lower()}.csv"  # Files are lowercase
            if not price_file.exists():
                logging.warning(f"Price data not found for {symbol}")
                continue
                
            # Read price data
            df = pd.read_csv(price_file)
            df['date'] = pd.to_datetime(df['timestamp']).dt.date  # Convert timestamp to date
            df.set_index('date', inplace=True)
            
            # Add closing price to main dataframe
            self.price_data[symbol] = df['close']
            
        logging.info(f"Loaded price data for {len(self.price_data.columns)} symbols")
        return self.price_data
        
    def calculate_portfolio_value(self):
        """Calculate daily portfolio value based on holdings and prices"""
        if self.price_data is None:
            self.load_price_data()
            
        # Align dates between holdings and prices
        common_dates = self.holdings.index.intersection(self.price_data.index)
        holdings = self.holdings.loc[common_dates]
        prices = self.price_data.loc[common_dates]
        
        # Calculate position values (quantity * price)
        position_values = holdings * prices
        
        # Calculate total portfolio value
        self.portfolio_value = pd.DataFrame({
            'total_value': position_values.sum(axis=1),
            'num_positions': (holdings != 0).sum(axis=1)
        })
        
        # Add position values
        for symbol in holdings.columns:
            if symbol in prices.columns:
                self.portfolio_value[f'{symbol}_value'] = position_values[symbol]
        
        logging.info(f"Calculated portfolio values from {common_dates.min()} to {common_dates.max()}")
        return self.portfolio_value
    
    def get_portfolio_summary(self, date=None):
        """Get portfolio summary for a specific date"""
        if self.portfolio_value is None:
            self.calculate_portfolio_value()
            
        if date is None:
            date = self.portfolio_value.index[-1]
            
        if date not in self.portfolio_value.index:
            closest_date = self.portfolio_value.index[self.portfolio_value.index <= date][-1]
            logging.warning(f"Date {date} not found, using closest date {closest_date}")
            date = closest_date
            
        summary = self.portfolio_value.loc[date].to_dict()
        
        # Add position details
        positions = []
        for col in self.portfolio_value.columns:
            if col.endswith('_value') and col != 'total_value':
                symbol = col[:-6]  # Remove '_value' suffix
                value = summary[col]
                if value > 0:
                    quantity = self.holdings.loc[date, symbol]
                    price = self.price_data.loc[date, symbol]
                    positions.append({
                        'symbol': symbol,
                        'quantity': quantity,
                        'price': price,
                        'value': value,
                        'weight': value / summary['total_value']
                    })
        
        summary['positions'] = sorted(positions, key=lambda x: x['value'], reverse=True)
        return summary

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize portfolio valuation
    portfolio = PortfolioValuation(
        holdings_file="data/processed/holdings.csv",
        data_directory="data"
    )
    
    # Calculate portfolio values
    portfolio_values = portfolio.calculate_portfolio_value()
    
    # Print some examples
    print("\nPortfolio Value Time Series:")
    print(portfolio_values[['total_value', 'num_positions']].tail())
    
    print("\nLatest Portfolio Summary:")
    summary = portfolio.get_portfolio_summary()
    print(f"Total Value: ₹{summary['total_value']:,.2f}")
    print(f"Number of Positions: {summary['num_positions']}")
    print("\nTop 5 Positions:")
    for pos in summary['positions'][:5]:
        print(f"{pos['symbol']}: {pos['quantity']} shares @ ₹{pos['price']:.2f} = ₹{pos['value']:,.2f} ({pos['weight']*100:.1f}%)")
