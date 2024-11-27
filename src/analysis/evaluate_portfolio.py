import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
import os
import sys
sys.path.append("../../")  # Add project root to path
from src.core.portfolio_manager import Portfolio
from src.visualization.portfolio_visualizer import PortfolioVisualizer

def calculate_portfolio_metrics(portfolio: Portfolio) -> dict:
    """Calculate comprehensive portfolio metrics"""
    metrics = {
        'Total Value': 'N/A',
        'Total Investment': 'N/A',
        'Total Return (%)': 'N/A',
        'Number of Positions': 0,
        'Annualized Return (%)': 'N/A',
        'Daily Volatility (%)': 'N/A',
        'Annualized Volatility (%)': 'N/A',
        'Sharpe Ratio': 'N/A',
        'Max Drawdown (%)': 'N/A',
        'Beta': 'N/A',
        'Alpha (%)': 'N/A',
        'Holdings': pd.DataFrame(),
        'Visualization Files': {}  
    }
    
    if portfolio.portfolio_value is None or portfolio.portfolio_value.empty:
        logging.error("Portfolio value DataFrame is None or empty")
        return metrics
    
    # First calculate holdings metrics to get accurate total value and investment
    holdings_df = calculate_holdings_metrics(portfolio)
    
    if not holdings_df.empty:
        # Calculate total value and investment from holdings
        total_value = holdings_df['Market Value'].sum()
        total_investment = holdings_df['Cost Basis'].sum()
        
        metrics['Total Value'] = total_value
        metrics['Total Investment'] = total_investment
        
        # Calculate total return using the actual totals
        if total_investment > 0:
            total_return = ((total_value - total_investment) / total_investment) * 100
            metrics['Total Return (%)'] = total_return
    
    total_value_series = portfolio.portfolio_value['total_value']
    logging.info(f"Total value series length: {len(total_value_series)}")
    logging.info(f"Total value series range: {total_value_series.min()} to {total_value_series.max()}")
    logging.info(f"Number of NaN values: {total_value_series.isna().sum()}")
    
    # Get latest portfolio value
    latest_value = total_value_series.iloc[-1]
    
    # Calculate annualized return
    if len(total_value_series) > 1:
        days = (total_value_series.index[-1] - total_value_series.index[0]).days
        if days > 0:
            total_return_decimal = (latest_value - metrics['Total Investment']) / metrics['Total Investment']
            annualized_return = ((1 + total_return_decimal) ** (365 / days) - 1) * 100
            metrics['Annualized Return (%)'] = annualized_return
    
    # Calculate Sharpe Ratio and Volatility
    if len(total_value_series) > 1:
        daily_returns = total_value_series.pct_change().dropna()
        if len(daily_returns) > 0:
            # Calculate daily volatility (standard deviation of returns)
            daily_vol = daily_returns.std() * 100  # Convert to percentage
            
            # Calculate annualized volatility
            # For daily data, multiply by sqrt(252) - number of trading days in a year
            annualized_vol = daily_vol * np.sqrt(252)
            
            # Only set metrics if we have reasonable values (less than 100% daily volatility)
            if daily_vol < 100:
                metrics['Daily Volatility (%)'] = daily_vol
                metrics['Annualized Volatility (%)'] = annualized_vol
                
                # Calculate Sharpe ratio using excess returns over risk-free rate
                risk_free_rate = 0.05  # 5% annual risk-free rate
                excess_return = (daily_returns.mean() * 252) - risk_free_rate  # Annualized excess return
                if annualized_vol > 0:
                    sharpe_ratio = excess_return / (annualized_vol / 100)  # Convert vol back to decimal
                    metrics['Sharpe Ratio'] = sharpe_ratio
            
            # Calculate maximum drawdown
            rolling_max = total_value_series.expanding().max()
            drawdown = ((total_value_series - rolling_max) / rolling_max) * 100
            max_drawdown = drawdown.min()
            if not np.isnan(max_drawdown):
                metrics['Max Drawdown (%)'] = max_drawdown
    
    # Calculate number of current positions
    if portfolio.holdings is not None:
        latest_holdings = portfolio.holdings.iloc[-1]
        current_positions = len(latest_holdings[latest_holdings > 0])
        metrics['Number of Positions'] = current_positions
    
    return metrics

def calculate_holdings_metrics(portfolio: Portfolio) -> pd.DataFrame:
    """Calculate detailed metrics for each holding"""
    if portfolio.holdings is None or portfolio.holdings.empty:
        return pd.DataFrame()
        
    # Get latest holdings
    latest_holdings = portfolio.holdings.iloc[-1]
    
    # Initialize holdings metrics
    holdings_data = []
    total_value = portfolio.portfolio_value['total_value'].iloc[-1]
    
    # Calculate metrics for each holding
    for symbol in latest_holdings.index:
        qty = latest_holdings[symbol]
        if qty <= 0:  # Skip positions that have been closed
            continue
            
        # Get current price
        current_price = portfolio.price_data[symbol].iloc[-1]
        
        # Calculate average buying price
        buy_trades = [t for t in portfolio.transactions 
                     if t.symbol.lower() == symbol.lower() and t.quantity > 0]
        total_cost = sum(t.quantity * t.price for t in buy_trades)
        total_qty = sum(t.quantity for t in buy_trades)
        avg_price = total_cost / total_qty if total_qty > 0 else 0
        
        # Calculate returns
        position_value = qty * current_price
        cost_basis = qty * avg_price
        absolute_return = position_value - cost_basis
        return_pct = (absolute_return / cost_basis * 100) if cost_basis > 0 else 0
        weight = (position_value / total_value * 100) if total_value > 0 else 0
        
        holdings_data.append({
            'Symbol': symbol.upper(),
            'Quantity': qty,
            'Avg Price': avg_price,
            'Current Price': current_price,
            'Market Value': position_value,
            'Cost Basis': cost_basis,
            'Absolute Return': absolute_return,
            'Return %': return_pct,
            'Weight %': weight
        })
    
    # Convert to DataFrame and sort by weight
    holdings_df = pd.DataFrame(holdings_data)
    if not holdings_df.empty:
        holdings_df = holdings_df.sort_values('Weight %', ascending=False)
        
        # Format numeric columns
        holdings_df['Avg Price'] = holdings_df['Avg Price'].round(2)
        holdings_df['Current Price'] = holdings_df['Current Price'].round(2)
        holdings_df['Market Value'] = holdings_df['Market Value'].round(2)
        holdings_df['Cost Basis'] = holdings_df['Cost Basis'].round(2)
        holdings_df['Absolute Return'] = holdings_df['Absolute Return'].round(2)
        holdings_df['Return %'] = holdings_df['Return %'].round(2)
        holdings_df['Weight %'] = holdings_df['Weight %'].round(2)
    
    return holdings_df

def format_metrics_output(metrics: dict) -> str:
    """Format metrics for display"""
    output = []
    
    # Add timestamp header
    current_time = datetime.now()
    output.append("=" * 50)
    output.append(f"Portfolio Evaluation Report")
    output.append(f"Generated on: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
    output.append("=" * 50)
    output.append("")  # Empty line for better readability
    
    output.append("Portfolio Metrics:")
    output.append("-----------------")
    
    # Format basic metrics
    if metrics['Total Value'] != 'N/A':
        output.append(f"Total Value: ₹{metrics['Total Value']:,.2f}")
    else:
        output.append(f"Total Value: {metrics['Total Value']}")
        
    if metrics['Total Investment'] != 'N/A':
        output.append(f"Total Investment: ₹{metrics['Total Investment']:,.2f}")
    else:
        output.append(f"Total Investment: {metrics['Total Investment']}")
        
    if metrics['Total Return (%)'] != 'N/A':
        output.append(f"Total Return: {metrics['Total Return (%)']:.2f}%")
    else:
        output.append(f"Total Return: {metrics['Total Return (%)']}")
        
    output.append(f"Number of Positions: {metrics['Number of Positions']}")
    
    if metrics['Annualized Return (%)'] != 'N/A':
        output.append(f"Annualized Return: {metrics['Annualized Return (%)']:.2f}%")
    else:
        output.append(f"Annualized Return: {metrics['Annualized Return (%)']}")
        
    if metrics['Daily Volatility (%)'] != 'N/A':
        output.append(f"Daily Volatility: {metrics['Daily Volatility (%)']:.2f}%")
    else:
        output.append(f"Daily Volatility: {metrics['Daily Volatility (%)']}")
        
    if metrics['Annualized Volatility (%)'] != 'N/A':
        output.append(f"Annualized Volatility: {metrics['Annualized Volatility (%)']:.2f}%")
    else:
        output.append(f"Annualized Volatility: {metrics['Annualized Volatility (%)']}")
        
    if metrics['Sharpe Ratio'] != 'N/A':
        output.append(f"Sharpe Ratio: {metrics['Sharpe Ratio']:.2f}")
    else:
        output.append(f"Sharpe Ratio: {metrics['Sharpe Ratio']}")
        
    if metrics['Max Drawdown (%)'] != 'N/A':
        output.append(f"Maximum Drawdown: {metrics['Max Drawdown (%)']:.2f}%")
    else:
        output.append(f"Maximum Drawdown: {metrics['Max Drawdown (%)']}")
    
    # Add detailed holdings information
    if not metrics['Holdings'].empty:
        output.append("\nHoldings Details:")
        output.append("-----------------")
        holdings_str = metrics['Holdings'].to_string(index=False)
        output.append(holdings_str)
        
        # Log holdings details
        logging.info("Current Portfolio Holdings:")
        for _, row in metrics['Holdings'].iterrows():
            holding_info = (
                f"Symbol: {row['Symbol']:12} "
                f"Qty: {row['Quantity']:8.2f} "
                f"Avg Price: ₹{row['Avg Price']:10,.2f} "
                f"Current: ₹{row['Current Price']:10,.2f} "
                f"Cost: ₹{row['Cost Basis']:12,.2f} "
                f"Value: ₹{row['Market Value']:12,.2f} "
                f"Weight: {row['Weight %']:6.2f}% "
                f"Return: {row['Return %']:7.2f}%"
            )
            logging.info(holding_info)
    
    return "\n".join(output)

def main():
    # Set up logging
    os.makedirs('logs', exist_ok=True)
    log_file = os.path.join('logs', 'portfolio_evaluation.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )
    
    if len(sys.argv) != 2:
        print("Usage: python evaluate_portfolio.py <transactions_file>")
        sys.exit(1)
    
    transactions_file = sys.argv[1]
    
    # Create logs and images directories if they don't exist
    os.makedirs('logs', exist_ok=True)
    os.makedirs('images', exist_ok=True)
    
    try:
        # Initialize and load portfolio
        portfolio = Portfolio()
        portfolio.load_transactions(transactions_file)
        
        # Build holdings and load price data
        portfolio.build_holdings()
        portfolio.load_price_data()
        
        # Calculate portfolio value
        portfolio.calculate_portfolio_value()
        
        # Debug portfolio value data
        if portfolio.portfolio_value is not None:
            logging.info(f"Portfolio value DataFrame shape: {portfolio.portfolio_value.shape}")
            logging.info(f"Portfolio value columns: {portfolio.portfolio_value.columns}")
            if 'total_value' in portfolio.portfolio_value.columns:
                total_value_series = portfolio.portfolio_value['total_value']
                logging.info(f"Total value series length: {len(total_value_series)}")
                logging.info(f"Total value series range: {total_value_series.min()} to {total_value_series.max()}")
                logging.info(f"Number of NaN values: {total_value_series.isna().sum()}")
            else:
                logging.error("'total_value' column not found in portfolio value DataFrame")
        else:
            logging.error("Portfolio value DataFrame is None")
        
        # Calculate metrics
        metrics = calculate_portfolio_metrics(portfolio)
        
        # Add holdings information
        metrics['Holdings'] = calculate_holdings_metrics(portfolio)
        
        # Create visualizations
        visualizer = PortfolioVisualizer()
        visualizer.set_file_prefix(os.path.basename(transactions_file))  # Set the transaction filename for unique image names
        
        # Get portfolio value series for visualization
        if portfolio.portfolio_value is not None:
            try:
                portfolio_values = portfolio.portfolio_value['total_value'].ffill()  # Forward fill any gaps
                if not portfolio_values.empty and not portfolio_values.isna().all():
                    logging.info("Generating portfolio visualizations...")
                    
                    # Generate and verify each visualization
                    equity_curve_file = visualizer.plot_equity_curve(portfolio_values, "Portfolio Equity Curve")
                    logging.info(f"Equity curve saved to: {equity_curve_file}")
                    
                    drawdown_file = visualizer.plot_drawdown(portfolio_values, "Portfolio Drawdown Analysis")
                    logging.info(f"Drawdown plot saved to: {drawdown_file}")
                    
                    heatmap_file = visualizer.create_monthly_returns_heatmap(portfolio_values, "Monthly Returns Heatmap")
                    logging.info(f"Monthly returns heatmap saved to: {heatmap_file}")
                    
                    metrics['Visualization Files'] = {
                        'Equity Curve': equity_curve_file,
                        'Drawdown': drawdown_file,
                        'Monthly Returns': heatmap_file
                    }
                    
                    # Verify files were created
                    for plot_type, file_path in metrics['Visualization Files'].items():
                        if os.path.exists(file_path):
                            logging.info(f"{plot_type} plot created successfully at {file_path}")
                        else:
                            logging.error(f"Failed to create {plot_type} plot at {file_path}")
                            
                    logging.info("Portfolio visualizations generated successfully")
                else:
                    logging.warning("Portfolio value series is empty or contains only NaN values")
            except Exception as e:
                logging.error(f"Error generating visualizations: {str(e)}")
                logging.error(f"Error details:", exc_info=True)
        else:
            logging.warning("Portfolio value data is not available for visualization")
        
        # Format output
        output = format_metrics_output(metrics)
        
        # Save output to log file
        log_filename = f"logs/portfolio_evaluation_{os.path.basename(transactions_file).replace('.csv', '')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(log_filename, 'w') as f:
            f.write(output)
        
        print(output)
        print(f"\nDetailed evaluation saved to: {log_filename}")
        
    except Exception as e:
        logging.error(f"Error evaluating portfolio: {str(e)}")
        logging.error("Error details:", exc_info=True)
        raise

if __name__ == "__main__":
    main()
