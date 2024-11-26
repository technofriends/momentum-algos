import pandas as pd
import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from typing import List, Dict
from download_stock_data import download_stock_data
import config
import traceback
import json
import os

# Enhanced logging format
logging.basicConfig(
    filename='error.log',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s\n%(debug_info)s\n'
)

class CustomAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        debug_info = kwargs.pop('debug_info', '')
        kwargs['extra'] = {'debug_info': debug_info}
        return msg, kwargs

logger = CustomAdapter(logging.getLogger(__name__), {})

# Global counters with thread-safe access
class DownloadStats:
    def __init__(self):
        self._lock = threading.Lock()
        self.success_count = 0
        self.error_count = 0
        self.total_processed = 0
        self.errors = []  # Store detailed error information
    
    def increment_success(self):
        with self._lock:
            self.success_count += 1
            self.total_processed += 1
    
    def increment_error(self, symbol: str, error_msg: str, debug_info: str = None):
        with self._lock:
            self.error_count += 1
            self.total_processed += 1
            self.errors.append({
                'symbol': symbol,
                'error': error_msg,
                'debug_info': debug_info,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            })
    
    def get_stats(self):
        with self._lock:
            return {
                'success': self.success_count,
                'error': self.error_count,
                'total': self.total_processed,
                'errors': self.errors
            }

def get_available_symbol_files():
    """
    Get a list of available CSV files containing symbols from the input_files directory
    """
    input_dir = 'input_files'
    # Create input_files directory if it doesn't exist
    os.makedirs(input_dir, exist_ok=True)
    
    # Get all CSV files from input_files directory
    files = []
    try:
        for file in os.listdir(input_dir):
            if file.endswith('.csv'):
                files.append(os.path.join(input_dir, file))
    except Exception as e:
        logger.error(f"Error reading input directory: {str(e)}")
        return []
    
    if not files:
        print("\nNo CSV files found in input_files directory!")
        print("Please add CSV files with stock symbols to the input_files directory.")
        return []
    
    return sorted(files)

def read_symbols_from_file(filename: str) -> List[str]:
    """
    Read symbols from a CSV file with enhanced error logging
    """
    try:
        df = pd.read_csv(filename)
        
        # Try common column names for symbols
        symbol_columns = ['Scrip', 'Symbol', 'SYMBOL', 'symbol', 'Ticker', 'TICKER', 'ticker']
        found_column = None
        
        for col in symbol_columns:
            if col in df.columns:
                found_column = col
                break
        
        if found_column is None:
            # If no standard column found, use the first column
            found_column = df.columns[0]
            logger.warning(f"No standard symbol column found in {filename}. Using first column: {found_column}")
        
        # Get the symbols and handle different formats
        raw_symbols = df[found_column].astype(str).str.strip().tolist()
        symbols = []
        
        for symbol in raw_symbols:
            if not symbol or str(symbol).lower() == 'nan':
                continue
                
            # Remove trailing comma if present
            symbol = symbol.rstrip(',')
            
            # Handle pipe-separated format (e.g., "Nifty 50|26000")
            if '|' in symbol:
                symbol = symbol.split('|')[0].strip()
            
            if symbol:  # Only add non-empty symbols
                symbols.append(symbol)
        
        logger.info(f"Successfully read {len(symbols)} symbols from {filename}")
        return symbols
        
    except Exception as e:
        debug_info = traceback.format_exc()
        logger.error(
            f"Error reading symbols from {filename}",
            debug_info=debug_info
        )
        return []

def get_user_file_selection() -> List[int]:
    """
    Present file options to user and get their selection
    Returns a list of selected file numbers
    """
    files = get_available_symbol_files()
    
    if not files:
        print("\nNo CSV files found in input_files directory!")
        print("Please add CSV files with stock symbols to the input_files directory.")
        return []
    
    while True:
        print("\nAvailable symbol files:")
        for i, file in enumerate(files, 1):
            # Get the number of symbols in the file
            symbols = read_symbols_from_file(file)
            filename = os.path.basename(file)
            print(f"{i}. {filename} ({len(symbols)} symbols)")
        
        print("\nEnter file numbers to process (comma-separated, e.g., '1,2')")
        print("Or press Enter to process all files")
        print("Or 'q' to quit")
        
        choice = input("\nYour choice: ").strip().lower()
        
        if choice == 'q':
            return []
        
        if choice == '':
            return list(range(1, len(files) + 1))
        
        try:
            selections = [int(x.strip()) for x in choice.split(',')]
            if all(1 <= x <= len(files) for x in selections):
                return selections
            else:
                print("\nError: Please enter valid file numbers!")
        except ValueError:
            print("\nError: Please enter numbers separated by commas!")

def download_with_retry(symbol: str, stats: DownloadStats) -> Dict:
    """
    Download data for a single symbol with retry logic and enhanced error logging
    """
    display_symbol = symbol  # Keep original symbol for display
    
    for attempt in range(max(1, config.MAX_RETRIES)):  # Ensure at least 1 attempt
        try:
            result = download_stock_data(symbol)
            if result:
                stats.increment_success()
                return {'symbol': symbol, 'status': 'success', 'file': result}
            else:
                error_msg = f"Download failed for symbol: {display_symbol} (Attempt {attempt + 1}/{max(1, config.MAX_RETRIES)})"
                if attempt < config.MAX_RETRIES - 1:
                    print(f"{error_msg} - Retrying...")
                    time.sleep(config.RETRY_DELAY)
                    continue
                
                stats.increment_error(
                    symbol=display_symbol,
                    error_msg=f"Download returned None - API request failed"
                )
                return {'symbol': symbol, 'status': 'error', 'error': 'API request failed'}
            
        except Exception as e:
            error_msg = f"Error downloading {display_symbol}: {str(e)}"
            debug_info = f"Traceback:\n{traceback.format_exc()}"
            
            if attempt < config.MAX_RETRIES - 1:
                print(f"{error_msg} - Retrying...")
                time.sleep(config.RETRY_DELAY)
                continue
                
            stats.increment_error(
                symbol=display_symbol,
                error_msg=error_msg,
                debug_info=debug_info
            )
            return {'symbol': symbol, 'status': 'error', 'error': str(e)}

def process_symbols(symbols: List[str]) -> None:
    """
    Process a list of symbols using a thread pool with enhanced logging
    """
    stats = DownloadStats()
    total_symbols = len(symbols)
    start_time = time.time()
    
    print(f"\nStarting download of {total_symbols} symbols using {config.MAX_WORKERS} workers")
    print(f"Progress will be reported every {config.PROGRESS_BATCH_SIZE} symbols")
    
    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
        future_to_symbol = {
            executor.submit(download_with_retry, symbol, stats): symbol 
            for symbol in symbols
        }
        
        for i, future in enumerate(as_completed(future_to_symbol), 1):
            symbol = future_to_symbol[future]
            try:
                result = future.result()
                
                if i % config.PROGRESS_BATCH_SIZE == 0:
                    current_stats = stats.get_stats()
                    print(f"\nProgress: {i}/{total_symbols} symbols processed")
                    print(f"Success: {current_stats['success']}, Errors: {current_stats['error']}")
                
            except Exception as e:
                error_msg = f"Unexpected error processing {symbol}: {str(e)}"
                debug_info = f"Traceback:\n{traceback.format_exc()}"
                logger.error(error_msg, debug_info=debug_info)
                stats.increment_error(symbol=symbol, error_msg=error_msg, debug_info=debug_info)
            
            time.sleep(config.DOWNLOAD_DELAY)
    
    # Print final summary with enhanced error reporting
    final_stats = stats.get_stats()
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("\nDownload Summary:")
    print(f"Total symbols processed: {total_symbols}")
    print(f"Successful downloads: {final_stats['success']}")
    print(f"Failed downloads: {final_stats['error']}")
    print(f"Total time elapsed: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"Average time per symbol: {elapsed_time/total_symbols:.2f} seconds")
    
    if final_stats['error'] > 0:
        print("\nError Summary:")
        for error in final_stats['errors'][:5]:  # Show first 5 errors in console
            print(f"- {error['symbol']}: {error['error']}")
        if len(final_stats['errors']) > 5:
            print(f"... and {len(final_stats['errors']) - 5} more errors")
        print("\nCheck error.log for complete error details")
    
    # Write detailed error report
    if final_stats['errors']:
        error_report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_symbols': total_symbols,
            'success_count': final_stats['success'],
            'error_count': final_stats['error'],
            'elapsed_time': elapsed_time,
            'errors': final_stats['errors']
        }
        
        with open('error_report.json', 'w') as f:
            json.dump(error_report, f, indent=2)
        print("\nDetailed error report written to error_report.json")

def batch_download_stocks(file_numbers=None):
    """
    Interactive stock data downloader with enhanced error handling
    
    Args:
        file_numbers (list, optional): List of file numbers to process. If None, will prompt user interactively.
    """
    try:
        if file_numbers is None:
            file_numbers = get_user_file_selection()
        
        files = get_available_symbol_files()
        all_symbols = []
        
        # Read symbols from selected files
        for file_num in file_numbers:
            filename = files[file_num - 1]  # Adjust index since list is 0-based
            print(f"\nReading symbols from {filename}...")
            symbols = read_symbols_from_file(filename)
            all_symbols.extend(symbols)
            print(f"Found {len(symbols)} symbols in {filename}")
        
        if not all_symbols:
            print("No symbols found in selected files.")
            return
        
        print(f"\nProcessing {len(all_symbols)} total symbols...")
        process_symbols(all_symbols)
        
    except Exception as e:
        error_msg = f"Error in batch processing: {str(e)}\n{traceback.format_exc()}"
        debug_info = f"File numbers: {file_numbers}"
        logger.error(error_msg, debug_info=debug_info)
        print(f"Error: {str(e)}")
        print("Check error.log for details")

if __name__ == "__main__":
    batch_download_stocks()
