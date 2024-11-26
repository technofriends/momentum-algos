import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple

# Set style for better-looking graphs
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.2)
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.labelcolor'] = '#333333'
plt.rcParams['text.color'] = '#333333'
plt.rcParams['xtick.color'] = '#333333'
plt.rcParams['ytick.color'] = '#333333'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

class PortfolioVisualizer:
    def __init__(self):
        self.images_dir = 'images'
        # Create images directory if it doesn't exist
        os.makedirs(self.images_dir, exist_ok=True)
        
        # Color scheme
        self.colors = {
            'primary': '#2E86C1',
            'secondary': '#E74C3C',
            'accent': '#2ECC71',
            'text': '#333333',
            'grid': '#CCCCCC'
        }
        
    def calculate_drawdown(self, equity_curve: pd.Series) -> Tuple[pd.Series, float, pd.Timestamp, pd.Timestamp]:
        """Calculate the drawdown series and maximum drawdown"""
        rolling_max = equity_curve.expanding().max()
        drawdown = equity_curve / rolling_max - 1
        max_drawdown = drawdown.min()
        end_idx = drawdown.idxmin()
        start_idx = equity_curve[:end_idx].idxmax()
        return drawdown, max_drawdown, start_idx, end_idx
    
    def plot_equity_curve(self, portfolio_values: pd.Series, title: str = "Portfolio Equity Curve"):
        """Plot the equity curve with key statistics"""
        # Ensure index is datetime
        if not isinstance(portfolio_values.index, pd.DatetimeIndex):
            portfolio_values.index = pd.to_datetime(portfolio_values.index)
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Calculate returns and statistics
        returns = portfolio_values.pct_change()
        total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0] - 1) * 100
        annualized_return = ((1 + total_return/100) ** (252/len(portfolio_values)) - 1) * 100
        volatility = returns.std() * np.sqrt(252) * 100
        sharpe_ratio = annualized_return / volatility if volatility != 0 else 0
        
        # Plot equity curve
        ax.plot(portfolio_values.index, portfolio_values, 
                color=self.colors['primary'], 
                linewidth=2.5,
                label='Portfolio Value')
        
        # Add statistics box
        stats_text = (
            f'Total Return: {total_return:,.1f}%\n'
            f'Annualized Return: {annualized_return:,.1f}%\n'
            f'Annualized Volatility: {volatility:,.1f}%\n'
            f'Sharpe Ratio: {sharpe_ratio:.2f}'
        )
        
        # Create statistics box with custom style
        bbox_props = dict(
            boxstyle='round,pad=0.5',
            facecolor='white',
            edgecolor=self.colors['grid'],
            alpha=0.9
        )
        
        ax.text(0.02, 0.98, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=bbox_props,
                fontsize=10,
                color=self.colors['text'])
        
        # Customize grid
        ax.grid(True, linestyle='--', alpha=0.7, color=self.colors['grid'])
        
        # Customize title and labels
        ax.set_title(title, pad=20, fontsize=14, fontweight='bold', color=self.colors['text'])
        ax.set_xlabel('Date', labelpad=10, fontsize=12, color=self.colors['text'])
        ax.set_ylabel('Value (INR)', labelpad=10, fontsize=12, color=self.colors['text'])
        
        # Format y-axis with comma separator
        def format_value(x, p):
            if x >= 1e6:
                return f'{x/1e6:,.1f}M'
            elif x >= 1e3:
                return f'{x/1e3:,.1f}K'
            else:
                return f'{x:,.0f}'
        
        ax.yaxis.set_major_formatter(plt.FuncFormatter(format_value))
        
        # Rotate x-axis labels
        plt.xticks(rotation=45)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the plot
        filename = os.path.join(self.images_dir, 'equity_curve.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return filename
    
    def plot_drawdown(self, portfolio_values: pd.Series, title: str = "Portfolio Drawdown Analysis"):
        """Plot the drawdown analysis"""
        # Ensure index is datetime
        if not isinstance(portfolio_values.index, pd.DatetimeIndex):
            portfolio_values.index = pd.to_datetime(portfolio_values.index)
            
        drawdown, max_drawdown, start_idx, end_idx = self.calculate_drawdown(portfolio_values)
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Plot drawdown
        ax.fill_between(drawdown.index, 
                       drawdown * 100, 
                       0, 
                       color=self.colors['secondary'],
                       alpha=0.3,
                       label='Drawdown')
        
        ax.plot(drawdown.index, 
                drawdown * 100, 
                color=self.colors['secondary'],
                linewidth=1.5)
        
        # Highlight maximum drawdown period
        ax.axvspan(start_idx, end_idx, 
                  color=self.colors['secondary'], 
                  alpha=0.1)
        
        # Add maximum drawdown information
        max_dd_text = (
            f'Maximum Drawdown: {max_drawdown*100:.1f}%\n'
            f'Start: {start_idx.strftime("%Y-%m-%d")}\n'
            f'End: {end_idx.strftime("%Y-%m-%d")}\n'
            f'Duration: {(end_idx - start_idx).days} days'
        )
        
        # Create info box with custom style
        bbox_props = dict(
            boxstyle='round,pad=0.5',
            facecolor='white',
            edgecolor=self.colors['grid'],
            alpha=0.9
        )
        
        ax.text(0.02, 0.02, max_dd_text,
                transform=ax.transAxes,
                verticalalignment='bottom',
                bbox=bbox_props,
                fontsize=10,
                color=self.colors['text'])
        
        # Customize grid
        ax.grid(True, linestyle='--', alpha=0.7, color=self.colors['grid'])
        
        # Customize title and labels
        ax.set_title(title, pad=20, fontsize=14, fontweight='bold', color=self.colors['text'])
        ax.set_xlabel('Date', labelpad=10, fontsize=12, color=self.colors['text'])
        ax.set_ylabel('Drawdown (%)', labelpad=10, fontsize=12, color=self.colors['text'])
        
        # Add zero line
        ax.axhline(y=0, color=self.colors['grid'], linestyle='-', linewidth=1)
        
        # Rotate x-axis labels
        plt.xticks(rotation=45)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the plot
        filename = os.path.join(self.images_dir, 'drawdown_analysis.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return filename

    def create_monthly_returns_heatmap(self, portfolio_values: pd.Series, title: str = "Monthly Returns Heatmap"):
        """Create a heatmap of monthly returns"""
        # Ensure index is datetime
        if not isinstance(portfolio_values.index, pd.DatetimeIndex):
            portfolio_values.index = pd.to_datetime(portfolio_values.index)
            
        # Calculate daily returns
        returns = portfolio_values.pct_change()
        
        # Create monthly returns
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        # Create a pivot table for the heatmap
        monthly_returns.index = pd.MultiIndex.from_arrays([
            monthly_returns.index.year,
            monthly_returns.index.month
        ])
        returns_table = monthly_returns.unstack() * 100
        
        # Set up the plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create custom colormap
        cmap = sns.diverging_palette(10, 133, as_cmap=True)
        
        # Create heatmap
        sns.heatmap(returns_table,
                   ax=ax,
                   annot=True,
                   fmt='.1f',
                   center=0,
                   cmap=cmap,
                   cbar_kws={'label': 'Monthly Return (%)'},
                   annot_kws={'size': 9},
                   linewidths=0.5,
                   linecolor='white')
        
        # Customize title and labels
        ax.set_title(title, pad=20, fontsize=14, fontweight='bold', color=self.colors['text'])
        ax.set_xlabel('Month', labelpad=10, fontsize=12, color=self.colors['text'])
        ax.set_ylabel('Year', labelpad=10, fontsize=12, color=self.colors['text'])
        
        # Replace month numbers with month names
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax.set_xticklabels([month_names[i-1] for i in returns_table.columns], rotation=0)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the plot
        filename = os.path.join(self.images_dir, 'monthly_returns_heatmap.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return filename

def main():
    """Example usage of the PortfolioVisualizer"""
    # Create sample data
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    
    # Simulate a portfolio with some realistic characteristics
    returns = np.random.normal(0.0004, 0.01, len(dates))  # Mean daily return ~10% annually, 16% volatility
    portfolio_values = 100000 * (1 + returns).cumprod()  # Start with 100,000
    portfolio_values = pd.Series(portfolio_values, index=dates)
    
    # Create visualizations
    visualizer = PortfolioVisualizer()
    
    equity_curve_file = visualizer.plot_equity_curve(portfolio_values)
    print(f"Equity curve saved as: {equity_curve_file}")
    
    drawdown_file = visualizer.plot_drawdown(portfolio_values)
    print(f"Drawdown analysis saved as: {drawdown_file}")
    
    heatmap_file = visualizer.create_monthly_returns_heatmap(portfolio_values)
    print(f"Monthly returns heatmap saved as: {heatmap_file}")

if __name__ == "__main__":
    main()
