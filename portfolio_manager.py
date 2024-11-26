import pandas as pd
from pathlib import Path
import logging
from typing import Optional, Dict, List
from dataclasses import dataclass
from datetime import datetime, date
import numpy as np

@dataclass
class Transaction:
    date: date
    symbol: str
    quantity: float
    price: float
    transaction_type: str = "TRADE"  # TRADE, SPLIT, DIVIDEND, etc.
    notes: str = ""

class Portfolio:
    def __init__(self, data_directory: str = "data"):
        """
        Initialize a new portfolio instance
        
        Args:
            data_directory: Directory containing price data files
        """
        self.transactions: List[Transaction] = []
        self.holdings: Optional[pd.DataFrame] = None
        self.portfolio_value: Optional[pd.DataFrame] = None
        self.data_directory = Path(data_directory)
        self.price_data: Optional[pd.DataFrame] = None
        
    def load_transactions(self, transaction_file: str) -> None:
        """Load transactions from a CSV file"""
        df = pd.read_csv(transaction_file)
        
        # Convert transactions to our internal format
        for _, row in df.iterrows():
            transaction = Transaction(
                date=pd.to_datetime(row['date']).date(),
                symbol=row['symbol'],
                quantity=float(row['qty']),
                price=float(row['price'])
            )
            self.transactions.append(transaction)
            
        self.transactions.sort(key=lambda x: x.date)
        logging.info(f"Loaded {len(self.transactions)} transactions")
        
    def build_holdings(self) -> pd.DataFrame:
        """Build holdings dataframe from transactions"""
        if not self.transactions:
            raise ValueError("No transactions loaded")
            
        # Get unique dates and symbols
        all_dates = pd.date_range(
            start=self.transactions[0].date,
            end=self.transactions[-1].date,
            freq='D'
        )
        symbols = list(set(t.symbol for t in self.transactions))
        
        # Initialize holdings DataFrame
        self.holdings = pd.DataFrame(0, index=all_dates, columns=symbols)
        
        # Process each transaction
        current_holdings = {symbol: 0 for symbol in symbols}
        for transaction in self.transactions:
            current_holdings[transaction.symbol] += transaction.quantity
            self.holdings.loc[transaction.date:, transaction.symbol] = current_holdings[transaction.symbol]
            
        # Convert index to date for consistency
        self.holdings.index = self.holdings.index.date
        logging.info(f"Built holdings for {len(symbols)} symbols from {all_dates[0].date()} to {all_dates[-1].date()}")
        return self.holdings
    
    def load_price_data(self) -> pd.DataFrame:
        """Load price data for all symbols in holdings"""
        if self.holdings is None:
            raise ValueError("No holdings data available. Call build_holdings() first")
            
        symbols = self.holdings.columns.tolist()
        self.price_data = pd.DataFrame()
        today = pd.Timestamp.today().date()
        
        for symbol in symbols:
            # Try different case variations and handle special characters
            possible_filenames = [
                f"{symbol.lower()}.csv",
                f"{symbol.upper()}.csv",
                f"{symbol}.csv"
            ]
            
            price_file = None
            for filename in possible_filenames:
                temp_file = self.data_directory / filename
                if temp_file.exists():
                    price_file = temp_file
                    break
                    
            if price_file is None:
                logging.warning(f"Price data not found for {symbol}")
                continue
                
            # Read price data
            df = pd.read_csv(price_file)
            df['date'] = pd.to_datetime(df['timestamp']).dt.date
            
            # Filter out future dates
            df = df[df['date'] <= today]
            
            df.set_index('date', inplace=True)
            
            # Add closing price to main dataframe
            self.price_data[symbol] = df['close']
            
        logging.info(f"Loaded price data for {len(self.price_data.columns)} symbols up to {today}")
        return self.price_data
    
    def calculate_portfolio_value(self) -> pd.DataFrame:
        """Calculate portfolio value based on holdings and prices"""
        if self.price_data is None:
            self.load_price_data()
            
        # Align dates between holdings and prices
        common_dates = self.holdings.index.intersection(self.price_data.index)
        holdings = self.holdings.loc[common_dates]
        prices = self.price_data.loc[common_dates]
        
        # Calculate position values (quantity * price)
        position_values = holdings * prices
        
        # Calculate total portfolio value and metrics
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
    
    def get_portfolio_summary(self, as_of_date: Optional[date] = None) -> Dict:
        """Get portfolio summary for a specific date"""
        if self.portfolio_value is None:
            self.calculate_portfolio_value()
            
        if as_of_date is None:
            as_of_date = self.portfolio_value.index[-1]
            
        if as_of_date not in self.portfolio_value.index:
            closest_date = self.portfolio_value.index[self.portfolio_value.index <= as_of_date][-1]
            logging.warning(f"Date {as_of_date} not found, using closest date {closest_date}")
            as_of_date = closest_date
            
        summary = self.portfolio_value.loc[as_of_date].to_dict()
        
        # Add position details
        positions = []
        for col in self.portfolio_value.columns:
            if col.endswith('_value') and col != 'total_value':
                symbol = col[:-6]  # Remove '_value' suffix
                value = summary[col]
                if value > 0:
                    quantity = self.holdings.loc[as_of_date, symbol]
                    price = self.price_data.loc[as_of_date, symbol]
                    positions.append({
                        'symbol': symbol,
                        'quantity': quantity,
                        'price': price,
                        'value': value,
                        'weight': value / summary['total_value']
                    })
        
        summary['positions'] = sorted(positions, key=lambda x: x['value'], reverse=True)
        return summary

    def calculate_statistics(self, benchmark_returns=None, risk_free_rate=0.05):
        """
        Calculate comprehensive portfolio statistics
        
        Args:
            benchmark_returns (pd.Series, optional): Daily returns of the benchmark
            risk_free_rate (float): Annual risk-free rate, defaults to 5%
        
        Returns:
            dict: Dictionary containing all calculated statistics
        """
        if self.portfolio_value is None:
            raise ValueError("Portfolio value not calculated yet")
            
        # Get daily returns
        daily_returns = self.portfolio_value['total_value'].pct_change()
        daily_returns = daily_returns.dropna()
        
        # 1. Return Metrics
        total_return = (self.portfolio_value['total_value'].iloc[-1] / self.portfolio_value['total_value'].iloc[0] - 1)
        days = (self.portfolio_value.index[-1] - self.portfolio_value.index[0]).days
        cagr = (1 + total_return) ** (365/days) - 1
        
        # Rolling returns
        if len(daily_returns) >= 252:  # If we have at least a year of data
            rolling_1y = (daily_returns + 1).rolling(252).agg(lambda x: x.prod()) - 1
        
        # 2. Risk Metrics
        daily_vol = daily_returns.std()
        annual_vol = daily_vol * np.sqrt(252)
        
        # Downside deviation (for Sortino)
        downside_returns = daily_returns[daily_returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252)
        
        # Maximum Drawdown
        cum_returns = (1 + daily_returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = cum_returns/rolling_max - 1
        max_drawdown = drawdowns.min()
        
        # Value at Risk (95%)
        var_95 = daily_returns.quantile(0.05)
        cvar_95 = daily_returns[daily_returns <= var_95].mean()
        
        # 3. Risk-Adjusted Returns
        daily_rf = (1 + risk_free_rate)**(1/252) - 1
        excess_returns = daily_returns - daily_rf
        
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / daily_vol
        sortino_ratio = np.sqrt(252) * excess_returns.mean() / downside_vol
        calmar_ratio = cagr / abs(max_drawdown)
        
        # 4. Market-Related Metrics (if benchmark provided)
        if benchmark_returns is not None:
            # Align benchmark returns with portfolio dates
            benchmark_returns = benchmark_returns[daily_returns.index]
            
            # Calculate Beta and Alpha
            covar = daily_returns.cov(benchmark_returns)
            benchmark_var = benchmark_returns.var()
            beta = covar / benchmark_var
            
            expected_return = daily_rf + beta * (benchmark_returns.mean() - daily_rf)
            alpha = daily_returns.mean() - expected_return
            
            # R-squared
            correlation = daily_returns.corr(benchmark_returns)
            r_squared = correlation ** 2
            
            # Tracking Error
            tracking_error = (daily_returns - benchmark_returns).std() * np.sqrt(252)
            
            # Information Ratio
            active_return = daily_returns.mean() - benchmark_returns.mean()
            information_ratio = np.sqrt(252) * active_return / tracking_error
        
        # 5. Portfolio Composition Metrics
        position_columns = [col for col in self.portfolio_value.columns if col.endswith('_value') and col != 'total_value']
        latest_weights = self.portfolio_value[position_columns].iloc[-1] / self.portfolio_value['total_value'].iloc[-1]
        concentration = (latest_weights ** 2).sum()  # Herfindahl-Hirschman Index
        
        # Calculate turnover
        if len(self.transactions) > 0:
            total_value = self.portfolio_value['total_value'].mean()
            turnover = sum(abs(t.quantity * t.price) for t in self.transactions) / total_value
        else:
            turnover = 0
            
        stats = {
            'return_metrics': {
                'total_return': total_return,
                'cagr': cagr,
                'latest_value': self.portfolio_value['total_value'].iloc[-1],
                'rolling_1y': rolling_1y.iloc[-1] if len(daily_returns) >= 252 else None
            },
            'risk_metrics': {
                'daily_volatility': daily_vol,
                'annual_volatility': annual_vol,
                'downside_volatility': downside_vol,
                'max_drawdown': max_drawdown,
                'var_95': var_95,
                'cvar_95': cvar_95
            },
            'risk_adjusted_returns': {
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio
            },
            'composition_metrics': {
                'num_positions': len(self.holdings.columns),
                'concentration': concentration,
                'turnover_ratio': turnover
            }
        }
        
        if benchmark_returns is not None:
            stats['market_metrics'] = {
                'beta': beta,
                'alpha': alpha * 252,  # Annualized
                'r_squared': r_squared,
                'tracking_error': tracking_error,
                'information_ratio': information_ratio
            }
            
        return stats
        
    def print_statistics(self, benchmark_returns=None, risk_free_rate=0.05):
        """Print formatted portfolio statistics"""
        stats = self.calculate_statistics(benchmark_returns, risk_free_rate)
        
        print("\n=== Portfolio Statistics ===")
        
        print("\nReturn Metrics:")
        print(f"Total Return: {stats['return_metrics']['total_return']*100:.1f}%")
        print(f"CAGR: {stats['return_metrics']['cagr']*100:.1f}%")
        if stats['return_metrics']['rolling_1y'] is not None:
            print(f"1-Year Rolling Return: {stats['return_metrics']['rolling_1y']*100:.1f}%")
        
        print("\nRisk Metrics:")
        print(f"Annual Volatility: {stats['risk_metrics']['annual_volatility']*100:.1f}%")
        print(f"Maximum Drawdown: {stats['risk_metrics']['max_drawdown']*100:.1f}%")
        print(f"95% VaR (daily): {stats['risk_metrics']['var_95']*100:.1f}%")
        
        print("\nRisk-Adjusted Returns:")
        print(f"Sharpe Ratio: {stats['risk_adjusted_returns']['sharpe_ratio']:.2f}")
        print(f"Sortino Ratio: {stats['risk_adjusted_returns']['sortino_ratio']:.2f}")
        print(f"Calmar Ratio: {stats['risk_adjusted_returns']['calmar_ratio']:.2f}")
        
        print("\nComposition Metrics:")
        print(f"Number of Positions: {stats['composition_metrics']['num_positions']}")
        print(f"Concentration (HHI): {stats['composition_metrics']['concentration']:.2f}")
        print(f"Turnover Ratio: {stats['composition_metrics']['turnover_ratio']:.2f}")
        
        if 'market_metrics' in stats:
            print("\nMarket Metrics:")
            print(f"Beta: {stats['market_metrics']['beta']:.2f}")
            print(f"Alpha (annual): {stats['market_metrics']['alpha']*100:.1f}%")
            print(f"R-squared: {stats['market_metrics']['r_squared']:.2f}")
            print(f"Information Ratio: {stats['market_metrics']['information_ratio']:.2f}")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create a new portfolio
    portfolio = Portfolio(data_directory="data")
    
    # Load transactions and build portfolio
    portfolio.load_transactions("sample_transactions.csv")
    portfolio.build_holdings()
    portfolio.calculate_portfolio_value()
    
    # Print portfolio summary
    print("\nPortfolio Value Time Series:")
    print(portfolio.portfolio_value[['total_value', 'num_positions']].tail())
    
    print("\nLatest Portfolio Summary:")
    summary = portfolio.get_portfolio_summary()
    print(f"Total Value: ₹{summary['total_value']:,.2f}")
    print(f"Number of Positions: {summary['num_positions']}")
    print("\nTop 5 Positions:")
    for pos in summary['positions'][:5]:
        print(f"{pos['symbol']}: {pos['quantity']} shares @ ₹{pos['price']:.2f} = ₹{pos['value']:,.2f} ({pos['weight']*100:.1f}%)")
    
    portfolio.print_statistics()
