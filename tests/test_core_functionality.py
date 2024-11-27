import unittest
import os
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch
from src.data.batch_download import batch_download_stocks
from src.analysis.evaluate_portfolio import calculate_portfolio_metrics
from src.core.portfolio_manager import Portfolio

class TestCoreFunctionality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set up test directories
        cls.test_data_dir = Path("data")
        cls.test_txn_dir = Path("transactions")
        cls.test_data_dir.mkdir(exist_ok=True)
        cls.test_txn_dir.mkdir(exist_ok=True)
        
        # Test transactions file
        cls.test_transactions_file = "n500_dynamic_txns.csv"
        
        # Create test transaction data if it doesn't exist
        if not (cls.test_txn_dir / cls.test_transactions_file).exists():
            # Create a simple test transaction file
            test_data = {
                'Entry Date': ['2023-01-01', '2023-02-01'],
                'Stock Name': ['RELIANCE', 'TCS'],
                'Qty': [100, 50],
                'Entry Price': [2500, 3500]
            }
            df = pd.DataFrame(test_data)
            df.to_csv(cls.test_txn_dir / cls.test_transactions_file, index=False)

    @patch('builtins.input', side_effect=['1', '2'])  # Mock user inputs for option 1 and 2
    def test_batch_download(self, mock_input):
        """Test batch download functionality with both options"""
        # Run batch download with file number 1
        result = batch_download_stocks(file_numbers=[1])
        
        # Verify some files were created in the data directory
        files = list(self.test_data_dir.glob("*.csv"))
        self.assertGreater(len(files), 0, "No files were downloaded")
        
        # Verify data content of downloaded files
        test_symbols = ['RELIANCE', 'TCS']  # Symbols from our test transaction file
        for symbol in test_symbols:
            file_path = self.test_data_dir / f"{symbol.lower()}.csv"
            self.assertTrue(file_path.exists(), f"Data file for {symbol} not found")
            
            df = pd.read_csv(file_path)
            self.assertGreater(len(df), 0, f"No data found for {symbol}")
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            self.assertTrue(all(col in df.columns for col in required_columns),
                          f"Missing required columns in {symbol} data")

    def test_portfolio_evaluation(self):
        """Test portfolio evaluation functionality"""
        # Create portfolio using correct directories
        portfolio = Portfolio(data_directory="data", txn_directory="transactions")
        portfolio.load_transactions(self.test_transactions_file)
        
        # Build holdings and load price data
        portfolio.build_holdings()
        portfolio.load_price_data()
        portfolio.calculate_portfolio_value()
        
        # Calculate metrics
        metrics = calculate_portfolio_metrics(portfolio)
        
        # Verify metrics structure and content
        self.assertIsInstance(metrics, dict, "Metrics should be returned as a dictionary")
        
        # Check for required metrics
        required_metrics = [
            'Total Value', 'Total Investment', 'Total Return (%)',
            'Annualized Return (%)', 'Sharpe Ratio'
        ]
        for metric in required_metrics:
            self.assertIn(metric, metrics, f"Missing metric: {metric}")

        # Only check numeric values for basic metrics that should always be available
        always_numeric_metrics = ['Total Value', 'Total Investment']
        for metric in always_numeric_metrics:
            self.assertIsInstance(metrics[metric], (int, float),
                                f"Metric {metric} should be numeric")

        # Verify basic metric relationships
        self.assertGreater(metrics['Total Value'], 0, "Total portfolio value should be positive")
        self.assertGreater(metrics['Total Investment'], 0, "Total investment should be positive")

if __name__ == '__main__':
    unittest.main()
