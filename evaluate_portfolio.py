import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
import os
from portfolio_manager import Portfolio
import sys

def calculate_portfolio_metrics(portfolio: Portfolio) -> dict:
    """Calculate comprehensive portfolio metrics"""
    metrics = {}
    
    # Get today's date
    today = pd.Timestamp.today().date()
    
    # Get holdings
    latest_holdings = portfolio.holdings.iloc[-1]
    
    # Calculate average buy price for each stock
    avg_buy_prices = {}
    total_investment = 0
    
    for symbol in latest_holdings.index:
        if latest_holdings[symbol] != 0:  # Only for current holdings
            symbol_txns = [t for t in portfolio.transactions if t.symbol == symbol]
            total_qty = 0
            total_cost = 0
            for txn in symbol_txns:
                if txn.quantity > 0:  # Only consider buy transactions
                    total_qty += txn.quantity
                    total_cost += txn.quantity * txn.price
            if total_qty > 0:
                avg_buy_prices[symbol] = total_cost / total_qty
                # Add to total investment if we still hold this stock
                if latest_holdings[symbol] > 0:
                    total_investment += latest_holdings[symbol] * avg_buy_prices[symbol]
    
    # Get the most recent price for each stock before today
    price_dates = portfolio.price_data.index[portfolio.price_data.index <= today]
    if len(price_dates) > 0:
        latest_date = price_dates[-1]  # Most recent date before or equal to today
        latest_prices = portfolio.price_data.loc[latest_date]
    else:
        latest_prices = pd.Series(np.nan, index=portfolio.price_data.columns)
    
    holdings_value = pd.DataFrame({
        'Quantity': latest_holdings[latest_holdings != 0],
        'Price': latest_prices,
        'Avg Buy Price': pd.Series(avg_buy_prices),
    })
    
    holdings_value['Value'] = holdings_value['Quantity'] * holdings_value['Price']
    holdings_value['Cost Basis'] = holdings_value['Quantity'] * holdings_value['Avg Buy Price']
    holdings_value['Return (%)'] = (holdings_value['Value'] / holdings_value['Cost Basis'] - 1) * 100
    
    # Calculate total value from actual holdings (excluding NaN values)
    total_value = holdings_value['Value'].dropna().sum()
    metrics['Total Value'] = f"₹{total_value:,.2f}"
    metrics['Total Investment'] = f"₹{total_investment:,.2f}"
    
    # Calculate total return
    total_return = ((total_value / total_investment) - 1) * 100 if total_investment > 0 else 0
    metrics['Total Return (%)'] = f"{total_return:.2f}%"
    
    # Recalculate weights based on actual total value
    holdings_value['Weight'] = holdings_value['Value'] / total_value * 100
    holdings_value = holdings_value.sort_values('Value', ascending=False)
    
    metrics['Number of Positions'] = len(holdings_value['Value'].dropna())
    metrics['Holdings'] = holdings_value
    
    # Performance metrics using actual portfolio value
    portfolio_daily_values = []
    common_dates = portfolio.holdings.index.intersection(portfolio.price_data.index)
    common_dates = [d for d in common_dates if d <= today]
    
    for date in common_dates:
        holdings = portfolio.holdings.loc[date]
        prices = portfolio.price_data.loc[date]
        daily_holdings = pd.DataFrame({
            'Quantity': holdings[holdings != 0],
            'Price': prices
        })
        daily_holdings['Value'] = daily_holdings['Quantity'] * daily_holdings['Price']
        portfolio_daily_values.append(daily_holdings['Value'].dropna().sum())
    
    portfolio_values = pd.Series(portfolio_daily_values, index=common_dates)
    portfolio_returns = portfolio_values.pct_change()
    
    # Calculate annualized return
    if len(portfolio_values) > 1:
        days = (portfolio_values.index[-1] - portfolio_values.index[0]).days
        if days > 0:
            annualized_return = (((total_value / total_investment) ** (365/days)) - 1) * 100
            metrics['Annualized Return (%)'] = f"{annualized_return:.2f}%"
        else:
            metrics['Annualized Return (%)'] = "N/A"
    else:
        metrics['Annualized Return (%)'] = "N/A"
    
    metrics['Daily Volatility (%)'] = f"{portfolio_returns.std() * 100:.2f}%"
    metrics['Annualized Volatility (%)'] = f"{portfolio_returns.std() * np.sqrt(252) * 100:.2f}%"
    
    # Risk metrics
    metrics['Sharpe Ratio'] = f"{(portfolio_returns.mean() / portfolio_returns.std()) * np.sqrt(252):.2f}"
    metrics['Max Drawdown (%)'] = f"{((portfolio_values / portfolio_values.cummax() - 1).min() * 100):.2f}%"
    
    # Concentration metrics
    metrics['Top 5 Holdings Weight (%)'] = f"{holdings_value['Weight'].dropna().head(5).sum():.2f}%"
    
    return metrics

def format_metrics_output(metrics: dict) -> str:
    """Format metrics into a readable string output"""
    output = []
    output.append("=" * 50)
    output.append("PORTFOLIO EVALUATION REPORT")
    output.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output.append("=" * 50)
    output.append("\n1. PORTFOLIO SUMMARY")
    output.append("-" * 30)
    output.append(f"Total Portfolio Value: {metrics['Total Value']}")
    output.append(f"Total Investment: {metrics['Total Investment']}")
    output.append(f"Number of Positions: {metrics['Number of Positions']}")
    
    output.append("\n2. PERFORMANCE METRICS")
    output.append("-" * 30)
    output.append(f"Total Return: {metrics['Total Return (%)']}")
    output.append(f"Annualized Return: {metrics['Annualized Return (%)']}")
    output.append(f"Daily Volatility: {metrics['Daily Volatility (%)']}")
    output.append(f"Annualized Volatility: {metrics['Annualized Volatility (%)']}")
    
    output.append("\n3. RISK METRICS")
    output.append("-" * 30)
    output.append(f"Sharpe Ratio: {metrics['Sharpe Ratio']}")
    output.append(f"Maximum Drawdown: {metrics['Max Drawdown (%)']}")
    
    output.append("\n4. CONCENTRATION METRICS")
    output.append("-" * 30)
    output.append(f"Top 5 Holdings Weight: {metrics['Top 5 Holdings Weight (%)']}")
    
    output.append("\n5. CURRENT HOLDINGS")
    output.append("-" * 30)
    holdings_df = metrics['Holdings']
    pd.set_option('display.float_format', lambda x: '{:,.2f}'.format(x))
    holdings_formatted = holdings_df[[
        'Quantity', 'Price', 'Avg Buy Price', 'Value', 'Return (%)', 'Weight'
    ]].round(2)
    output.append(holdings_formatted.to_string())
    
    return "\n".join(output)

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO, filename='logs/portfolio_evaluation.log', filemode='w')
    
    if len(sys.argv) != 2:
        print("Usage: python evaluate_portfolio.py <transactions_file>")
        sys.exit(1)
    
    transactions_file = sys.argv[1]
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    try:
        # Initialize and load portfolio
        portfolio = Portfolio()
        portfolio.load_transactions(transactions_file)
        
        # Build holdings and load price data
        portfolio.build_holdings()
        portfolio.load_price_data()
        
        # Calculate portfolio value
        portfolio.calculate_portfolio_value()
        
        # Calculate metrics
        metrics = calculate_portfolio_metrics(portfolio)
        
        # Format output
        output = format_metrics_output(metrics)
        
        # Print to console
        print(output)
        
        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = os.path.splitext(os.path.basename(transactions_file))[0]
        output_file = f"logs/portfolio_evaluation_{base_filename}_{timestamp}.txt"
        
        with open(output_file, 'w') as f:
            f.write(output)
        
        print(f"\nReport saved to: {output_file}")
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
