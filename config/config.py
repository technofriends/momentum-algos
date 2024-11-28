"""
Configuration settings for the stock data downloader
"""

# Number of worker threads for parallel downloads
MAX_WORKERS = 50

# Delay between downloads (in seconds) for each worker
DOWNLOAD_DELAY = 0

# Maximum retries for failed downloads
MAX_RETRIES = 3

# Delay between retries (in seconds)
RETRY_DELAY = 2.0

# Batch size for progress reporting
PROGRESS_BATCH_SIZE = 10
