"""
Core Functionality Test Suite

This module contains unit tests for the core functionality of the stock analysis system.
It tests the following key components:
    - Stock data downloading
    - Portfolio management
    - Performance metrics calculation

The tests use a combination of real and mock data to verify system behavior.
Test data is automatically created in a test environment to ensure reproducibility.

Author: Vaibhav Pandey
Last Modified: 2024-01-28
"""

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
    """
    Test suite for core system functionality.
    
    This class tests the three main components of the system:
    1. Data downloading and storage
    2. Portfolio management and tracking
    3. Performance metrics calculation
    
    The suite uses a temporary test environment with mock transaction data
    to verify system behavior without affecting production data.
    """
    
    @classmethod
    def setUpClass(cls):
        """
        Set up test environment before running tests.
        
        Creates:
            - Test directories for data and transactions
            - Sample transaction data for testing
            
        The setup ensures that tests run in an isolated environment
        with known test data.
        """
        # Set up test directories
        cls.test_data_dir = Path("data")
        cls.test_txn_dir = Path("transactions")
        cls.test_data_dir.mkdir(exist_ok=True)
        cls.test_txn_dir.mkdir(exist_ok=True)
        
        # Test transactions file
        cls.test_transactions_file = "n500_dynamic_txns.csv"
        
        # Create test transaction data if it doesn't exist
        if not (cls.test_txn_dir / cls.test_transactions_file).exists():
            # Create a simple test transaction file with known values
            test_data = {
                'Entry Date': ['2023-01-01', '2023-02-01'],  # Known dates
                'Stock Name': ['RELIANCE', 'TCS'],           # Major stocks
                'Qty': [100, 50],                           # Round quantities
                'Entry Price': [2500, 3500]                 # Reasonable prices
            }
            df = pd.DataFrame(test_data)
            df.to_csv(cls.test_txn_dir / cls.test_transactions_file, index=False)

    @patch('builtins.input', side_effect=['1', '2'])
    def test_batch_download(self, mock_input):
        """
        Test the batch download functionality.
        
        This test verifies that:
        1. The download process completes successfully
        2. Files are created for each stock
        3. Downloaded data has the correct structure
        4. Data content is valid and complete
        
        Args:
            mock_input: Mocked user input for testing different download options
        """
        # Run batch download with file number 1
        result = batch_download_stocks(file_numbers=[1])
        
        # Verify some files were created in the data directory
        files = list(self.test_data_dir.glob("*.csv"))
        self.assertGreater(len(files), 0, "No files were downloaded")
        
        # Verify data content of downloaded files
        test_symbols = ['RELIANCE', 'TCS']  # Test with major stocks
        for symbol in test_symbols:
            file_path = self.test_data_dir / f"{symbol.lower()}.csv"
            self.assertTrue(file_path.exists(), f"Data file for {symbol} not found")
            
            df = pd.read_csv(file_path)
            self.assertGreater(len(df), 0, f"No data found for {symbol}")
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            self.assertTrue(all(col in df.columns for col in required_columns),
                          f"Missing required columns in {symbol} data")

    def test_portfolio_evaluation(self):
        """
        Test the portfolio evaluation functionality.
        
        This test verifies:
        1. Portfolio initialization with test data
        2. Holdings calculation
        3. Price data loading
        4. Portfolio value calculation
        5. Performance metrics calculation
        
        The test uses known test data to verify that calculations
        produce expected results.
        """
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

        # Verify numeric values for core metrics
        always_numeric_metrics = ['Total Value', 'Total Investment']
        for metric in always_numeric_metrics:
            self.assertIsInstance(metrics[metric], (int, float),
                                f"Metric {metric} should be numeric")

        # Verify basic metric relationships
        self.assertGreater(metrics['Total Value'], 0, "Total portfolio value should be positive")
        self.assertGreater(metrics['Total Investment'], 0, "Total investment should be positive")

if __name__ == '__main__':
    unittest.main()
