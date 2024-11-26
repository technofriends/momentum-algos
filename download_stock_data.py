import os
import json
import requests
import logging
from datetime import datetime
import pandas as pd
import traceback

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

def format_request_debug_info(url, headers, data, response=None):
    """Format debug information for API requests"""
    debug_info = [
        "Debug Information:",
        f"URL: {url}",
        f"Request Headers: {json.dumps(headers, indent=2)}",
        f"Request Data: {json.dumps(data, indent=2)}"
    ]
    
    if response:
        debug_info.extend([
            f"Response Status Code: {response.status_code}",
            f"Response Headers: {json.dumps(dict(response.headers), indent=2)}",
            "Response Content (truncated):"
        ])
        try:
            content = response.json()
            debug_info.append(json.dumps(content, indent=2)[:500] + "...")
        except:
            debug_info.append(response.text[:500] + "...")
    
    return "\n".join(debug_info)

def download_stock_data(symbol):
    """
    Download stock data for a given symbol and save it to the data folder
    
    Args:
        symbol (str): Stock symbol to download data for
    
    Returns:
        str: Path to the saved file
    """
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # API endpoint
    url = 'https://tradepoint.definedge.com/tradepoint/data/rates'
    
    # Headers required for the request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:131.0) Gecko/20100101 Firefox/131.0',
        'Accept': '*/*',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Content-type': 'application/x-www-form-urlencoded',
        'Origin': 'https://tradepoint.definedge.com',
        'Connection': 'keep-alive',
        'Referer': 'https://tradepoint.definedge.com/index.html',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-origin',
        'Priority': 'u=0',
        'TE': 'trailers'
    }
    
    # Extract actual symbol if it contains a pipe
    display_symbol = symbol.strip()  # Remove any whitespace
    request_symbol = display_symbol  # Keep the full symbol as is - it already includes the ID
    
    # Form data
    data = {
        'exchange': 'NSE',
        'symbol': request_symbol  # Use the symbol as is, no need to append |-1
    }
    
    try:
        # Make POST request
        response = requests.post(url, headers=headers, data=data)
        
        # Log response details for debugging
        debug_info = format_request_debug_info(url, headers, data, response)
        debug_info += f"\nResponse Status: {response.status_code}"
        debug_info += f"\nResponse Content: {response.text}"
        
        # Check response status
        if response.status_code != 200:
            logger.error(
                f"HTTP {response.status_code} error for symbol: {display_symbol} (ID: {request_symbol})",
                debug_info=debug_info
            )
            return None
            
        response.raise_for_status()
        
        # Check if response is empty
        if not response.text:
            logger.error(
                f"Empty response received for symbol: {display_symbol} (ID: {request_symbol})",
                debug_info=debug_info
            )
            return None
            
        try:
            stock_data = response.json()
        except requests.exceptions.JSONDecodeError as e:
            logger.error(
                f"Invalid JSON response for symbol: {display_symbol} (ID: {request_symbol})",
                debug_info=debug_info + f"\nJSON Error: {str(e)}"
            )
            return None
        
        # Check if data is empty or only contains header
        if not stock_data.get('data'):
            logger.error(
                f"Empty data received for symbol: {display_symbol} (ID: {request_symbol})",
                debug_info=debug_info
            )
            return None
            
        # Convert to DataFrame
        df = pd.DataFrame(stock_data['data'], 
                         columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'])
        
        # Check if DataFrame is empty or contains only NaN values
        if df.empty or df.isnull().all().all():
            logger.error(
                f"Empty or invalid DataFrame for symbol: {display_symbol} (ID: {request_symbol})",
                debug_info=debug_info + f"\nDataFrame Info:\n{df.info()}"
            )
            return None
            
        # Check if any required columns are entirely empty
        empty_columns = df.columns[df.isnull().all()].tolist()
        if empty_columns:
            logger.error(
                f"Empty columns found for symbol {display_symbol} (ID: {request_symbol}): {', '.join(empty_columns)}",
                debug_info=debug_info + f"\nDataFrame Info:\n{df.info()}"
            )
            return None
            
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Save to CSV
        output_file = os.path.join('data', f'{request_symbol.lower()}.csv')
        df.to_csv(output_file, index=False)
        print(f"Data saved to {output_file}")
        
        return output_file
        
    except requests.exceptions.RequestException as e:
        debug_info = format_request_debug_info(url, headers, data)
        logger.error(
            f"Request error for symbol {display_symbol} (ID: {request_symbol}): {str(e)}",
            debug_info=debug_info + f"\nTraceback:\n{traceback.format_exc()}"
        )
        return None
    except Exception as e:
        debug_info = format_request_debug_info(url, headers, data)
        logger.error(
            f"Unexpected error for symbol {display_symbol} (ID: {request_symbol}): {str(e)}",
            debug_info=debug_info + f"\nTraceback:\n{traceback.format_exc()}"
        )
        return None

if __name__ == "__main__":
    # Example usage
    symbol = "ABREL"  # You can change this to any symbol
    download_stock_data(symbol)
