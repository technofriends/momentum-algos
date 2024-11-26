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

def get_available_symbol_files() -> Dict[int, str]:
    """
    Get a list of available CSV files containing symbols
    """
    files = {
        1: 'nse_total_market.csv',
        2: 'other_symbols.csv'
    }
    return files

def read_symbols_from_file(filename: str) -> List[str]:
    """
    Read symbols from a CSV file with enhanced error logging
    """
    try:
        df = pd.read_csv(filename)
        
        # Log file structure for debugging
        debug_info = (
            f"File Structure:\n"
            f"Columns: {df.columns.tolist()}\n"
            f"Shape: {df.shape}\n"
            f"First few rows:\n{df.head().to_string()}"
        )
        
        # Check if 'Scrip' column exists
        if 'Scrip' in df.columns:
            symbols = df['Scrip'].tolist()
        else:
            # Use the first column
            logger.error(
                f"'Scrip' column not found in {filename}, using first column: {df.columns[0]}",
                debug_info=debug_info
            )
            symbols = df.iloc[:, 0].tolist()
        
        # Remove any empty or NaN values
        symbols = [str(s).strip() for s in symbols if pd.notna(s) and str(s).strip()]
        
        if not symbols:
            logger.error(
                f"No valid symbols found in {filename}",
                debug_info=debug_info
            )
            return []
            
        return symbols
        
    except Exception as e:
        debug_info = f"Traceback:\n{traceback.format_exc()}"
        logger.error(
            f"Error reading file {filename}: {str(e)}",
            debug_info=debug_info
        )
        return []

def get_user_file_selection() -> List[int]:
    """
    Present file options to user and get their selection
    Returns a list of selected file numbers
    """
    files = get_available_symbol_files()
    
    while True:
        print("\nAvailable symbol files:")
        for num, filename in files.items():
            print(f"{num}. {filename}")
        
        print("\nEnter file number(s) to process (examples: '1' or '2' or '1,2'):")
        selection = input("> ").strip()
        
        try:
            # Handle comma-separated input
            if ',' in selection:
                selected_numbers = [int(x.strip()) for x in selection.split(',')]
            else:
                selected_numbers = [int(selection)]
            
            # Validate selections
            if all(num in files for num in selected_numbers):
                return selected_numbers
            else:
                print("Invalid selection. Please enter valid file numbers.")
        except ValueError:
            print("Invalid input. Please enter numbers only.")

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
            filename = files[file_num]
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
    # Show file selection menu
    batch_download_stocks()