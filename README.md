# Stock Analysis - Momentum Strategy

A comprehensive Python-based stock analysis system that implements and analyzes momentum-based trading strategies. The system provides tools for data collection, portfolio management, performance analysis, and visualization.

## 🌟 Key Features

- **Data Management**
  - Automated stock data downloading from TradePoint API
  - Robust error handling and validation
  - Support for both batch and single-symbol downloads
  
- **Portfolio Management**
  - Transaction tracking and portfolio state management
  - Support for multiple transaction types (buy, sell)
  - Historical portfolio value calculation
  
- **Performance Analysis**
  - Comprehensive portfolio metrics calculation
    - Total Value and Investment
    - Total and Annualized Returns
    - Daily Volatility
    - Sharpe Ratio
    - Maximum Drawdown
  - Time-series performance tracking
  
- **Visualization**
  - Equity curves
  - Drawdown analysis
  - Monthly return heatmaps
  - Performance metric dashboards

## 🏗 Architecture

The project follows a modular architecture designed for extensibility and maintainability:

```
momentum/
├── src/                    # Source code
│   ├── core/              # Core functionality
│   │   └── portfolio.py   # Portfolio management classes
│   ├── data/              # Data handling
│   │   └── download_stock_data.py  # Stock data downloading
│   ├── analysis/          # Analysis modules
│   │   └── evaluate_portfolio.py   # Portfolio evaluation
│   └── visualization/     # Visualization modules
│       └── visualizer.py  # Portfolio visualization
├── data/                  # Downloaded stock data
├── tests/                 # Test files
├── logs/                  # Log files and error tracking
├── config/               # Configuration
│   └── config.py         # System configuration
├── notebooks/            # Analysis notebooks
├── output/              # Generated reports
├── transactions/       # Transaction records
└── requirements.txt    # Project dependencies
```

### Component Details

#### Core (src/core/)
- `portfolio.py`: Implements the Portfolio class for managing stock holdings and transactions
  - Tracks individual transactions
  - Calculates current holdings
  - Manages portfolio state

#### Data (src/data/)
- `download_stock_data.py`: Handles stock data retrieval
  - Connects to TradePoint API
  - Downloads OHLCV data
  - Validates and processes raw data
  - Saves standardized CSV files

#### Analysis (src/analysis/)
- `evaluate_portfolio.py`: Performs portfolio analysis
  - Calculates performance metrics
  - Generates performance reports
  - Tracks historical performance

#### Visualization (src/visualization/)
- `visualizer.py`: Creates portfolio visualizations
  - Generates equity curves
  - Plots drawdown analysis
  - Creates return heatmaps

## 🚀 Setup and Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd momentum
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure settings:
- Copy `config/config.example.py` to `config/config.py`
- Update settings as needed

## 📊 Usage

### Downloading Stock Data

Single stock:
```bash
python -m src.data.download_stock_data SYMBOL
```

### Portfolio Analysis

Run portfolio evaluation:
```bash
python -m src.analysis.evaluate_portfolio
```

The analysis will:
1. Load transaction data
2. Download required stock data
3. Calculate portfolio metrics
4. Generate visualizations
5. Save reports to the output directory

## 📝 Configuration

Key configuration options in `config/config.py`:
- Data download settings
- Portfolio analysis parameters
- Visualization preferences
- Logging configuration

## 📈 Performance Metrics

The system calculates:
- **Total Value**: Current portfolio value
- **Total Investment**: Sum of all investments
- **Total Return**: Absolute return percentage
- **Annualized Return**: Return normalized to yearly basis
- **Daily Volatility**: Standard deviation of daily returns
- **Sharpe Ratio**: Risk-adjusted return metric
- **Maximum Drawdown**: Largest peak-to-trough decline

## 🔍 Logging and Error Handling

- Comprehensive error logging in `error.log`
- Detailed debug information for API requests
- Transaction validation logging
- Performance calculation tracking

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
