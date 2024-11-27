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
    
    total_value_series = portfolio.portfolio_value['total_value']
    logging.info(f"Total value series length: {len(total_value_series)}")
    logging.info(f"Total value series range: {total_value_series.min()} to {total_value_series.max()}")
    logging.info(f"Number of NaN values: {total_value_series.isna().sum()}")
    
    # Get latest portfolio value
    latest_value = total_value_series.iloc[-1]
    metrics['Total Value'] = latest_value
    
    # Calculate total investment
    total_investment = 0
    for txn in portfolio.transactions:
        if txn.transaction_type == 'TRADE':
            total_investment += txn.quantity * txn.price
    metrics['Total Investment'] = total_investment
    
    # Calculate total return
    if total_investment > 0:
        total_return = ((latest_value - total_investment) / total_investment) * 100
        metrics['Total Return (%)'] = total_return
    
    # Calculate annualized return
    if len(total_value_series) > 1:
        days = (total_value_series.index[-1] - total_value_series.index[0]).days
        if days > 0:
            total_return_decimal = (latest_value - total_investment) / total_investment
            annualized_return = ((1 + total_return_decimal) ** (365 / days) - 1) * 100
            metrics['Annualized Return (%)'] = annualized_return
    
    # Calculate Sharpe Ratio
    if len(total_value_series) > 1:
        daily_returns = total_value_series.pct_change().dropna()
        if len(daily_returns) > 0:
            volatility = daily_returns.std() * np.sqrt(252)  # Annualized volatility
            if volatility > 0:
                avg_return = daily_returns.mean() * 252  # Annualized return
                risk_free_rate = 0.05  # 5% annual risk-free rate
                sharpe_ratio = (avg_return - risk_free_rate) / volatility
                metrics['Sharpe Ratio'] = sharpe_ratio
    
    return metrics

def format_metrics_output(metrics: dict) -> str:
    """Format metrics for display"""
    output = []
    output.append("Portfolio Metrics:")
    output.append("-----------------")
    
    # Basic metrics
    if 'Total Value' in metrics:
        output.append(f"Total Value: ₹{metrics['Total Value']:,.2f}")
    if 'Total Investment' in metrics:
        output.append(f"Total Investment: ₹{metrics['Total Investment']:,.2f}")
    if 'Total Return' in metrics:
        output.append(f"Total Return: {metrics['Total Return']:.2f}%")
    if 'Number of Positions' in metrics:
        output.append(f"Number of Positions: {metrics['Number of Positions']}")
    
    # Performance metrics
    if 'Annualized Return' in metrics and metrics['Annualized Return'] != 'N/A':
        output.append(f"Annualized Return: {metrics['Annualized Return']:.2f}%")
    else:
        output.append("Annualized Return: N/A")
        
    if 'Sharpe Ratio' in metrics and metrics['Sharpe Ratio'] != 'N/A':
        output.append(f"Sharpe Ratio: {metrics['Sharpe Ratio']:.2f}")
    else:
        output.append("Sharpe Ratio: N/A")
        
    if 'Max Drawdown' in metrics and metrics['Max Drawdown'] != 'N/A':
        output.append(f"Maximum Drawdown: {metrics['Max Drawdown']:.2f}%")
    else:
        output.append("Maximum Drawdown: N/A")
        
    if 'Beta' in metrics and metrics['Beta'] != 'N/A':
        output.append(f"Beta: {metrics['Beta']:.2f}")
    else:
        output.append("Beta: N/A")
        
    if 'Alpha' in metrics and metrics['Alpha'] != 'N/A':
        output.append(f"Alpha: {metrics['Alpha']:.2f}%")
    else:
        output.append("Alpha: N/A")
    
    # Holdings table
    if 'Holdings' in metrics and not metrics['Holdings'].empty:
        output.append("\nCurrent Holdings:")
        output.append("----------------")
        holdings_str = metrics['Holdings'].to_string()
        output.append(holdings_str)
    
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
