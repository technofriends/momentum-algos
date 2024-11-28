"""
Configuration Module

This module contains all the configuration settings for the stock analysis system.
It centralizes important parameters that control the behavior of data downloading,
analysis, and visualization components.

Configuration Categories:
    - Download Settings: Controls parallel processing and rate limiting
    - Retry Settings: Defines behavior for handling failed downloads
    - Progress Settings: Controls how progress is reported
    - Analysis Settings: Parameters for portfolio analysis
    - Visualization Settings: Controls plot appearance and behavior
"""

#------------------------------------------------------------------------------
# Download Settings
#------------------------------------------------------------------------------

# Number of worker threads for parallel downloads
# Higher values increase download speed but may trigger rate limits
MAX_WORKERS = 50

# Delay between downloads (in seconds) for each worker
# Increase this if experiencing rate limiting issues
DOWNLOAD_DELAY = 0

#------------------------------------------------------------------------------
# Retry Settings
#------------------------------------------------------------------------------

# Maximum number of retry attempts for failed downloads
# System will attempt to download data this many times before giving up
MAX_RETRIES = 3

# Delay between retry attempts (in seconds)
# Exponential backoff is applied to this value for each retry
RETRY_DELAY = 2.0

#------------------------------------------------------------------------------
# Progress Settings
#------------------------------------------------------------------------------

# Number of downloads to process before reporting progress
# Lower values provide more frequent updates but may impact performance
PROGRESS_BATCH_SIZE = 10

#------------------------------------------------------------------------------
# Analysis Settings
#------------------------------------------------------------------------------

# Risk-free rate for Sharpe Ratio calculation (as decimal)
RISK_FREE_RATE = 0.05

# Minimum required data points for statistical calculations
MIN_DATA_POINTS = 252  # Approximately one trading year

# Confidence level for statistical calculations (as decimal)
CONFIDENCE_LEVEL = 0.95

#------------------------------------------------------------------------------
# Visualization Settings
#------------------------------------------------------------------------------

# Default figure size for plots (width, height in inches)
FIGURE_SIZE = (12, 8)

# Color scheme for plots
COLORS = {
    'equity_line': '#1f77b4',     # Blue
    'drawdown': '#d62728',        # Red
    'positive_return': '#2ca02c',  # Green
    'negative_return': '#ff7f0e',  # Orange
}

# Date format for plot labels
DATE_FORMAT = '%Y-%m-%d'

#------------------------------------------------------------------------------
# Logging Settings
#------------------------------------------------------------------------------

# Log file path
LOG_FILE = 'logs/error.log'

# Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_LEVEL = 'ERROR'

# Maximum log file size in bytes (10 MB)
MAX_LOG_SIZE = 10 * 1024 * 1024

# Number of backup log files to keep
LOG_BACKUP_COUNT = 5
