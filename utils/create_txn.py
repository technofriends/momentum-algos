"""
Transaction File Creator

This utility module converts trading data from a TSV format to a standardized CSV format
used by the portfolio analysis system. It handles both entry and exit transactions,
ensuring they are properly formatted and chronologically ordered.

Input TSV Format:
    Sr No | Stock Name | Entry Price | Exit Price | Qty | PnL | Entry Date | Exit Date

Output CSV Format:
    date,qty,symbol,price

Features:
    - Parses dates in '01 January 2023' format to 'YYYY-MM-DD'
    - Creates separate entry and exit transactions
    - Sorts transactions chronologically
    - Validates numeric fields
    - Handles missing exit dates gracefully

Author: Vaibhav Pandey
Last Modified: 2024-01-28
"""

import csv
from datetime import datetime
from typing import Tuple, List

def parse_date(date_str: str) -> str:
    """
    Convert date from '01 January 2023' format to 'YYYY-MM-DD'.
    
    Args:
        date_str: Date string in format 'DD Month YYYY'
        
    Returns:
        str: Date in 'YYYY-MM-DD' format
        
    Example:
        >>> parse_date('01 January 2023')
        '2023-01-01'
    """
    return datetime.strptime(date_str, '%d %B %Y').strftime('%Y-%m-%d')

def process_data(input_file: str, output_file: str) -> None:
    """
    Process trading data from TSV to CSV format.
    
    This function:
    1. Reads trading data from a TSV file
    2. Creates entry and exit transactions
    3. Sorts transactions chronologically
    4. Writes standardized CSV output
    
    Args:
        input_file: Path to input TSV file
        output_file: Path for output CSV file
        
    Example TSV row:
        1    RELIANCE    2500    2600    100    10000    01 January 2023    02 February 2023
    
    Generated transactions:
        2023-01-01,100,RELIANCE,2500.0
        2023-02-02,-100,RELIANCE,2600.0
    """
    transactions: List[Tuple[str, int, str, float]] = []
    
    # Process input TSV file
    with open(input_file, 'r') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        next(reader)  # Skip header row
        
        for row in reader:
            # Ensure row has minimum required fields
            if len(row) >= 7:
                sr_no, stock_name, entry_price, exit_price, qty, pnl, entry_date = row[:7]
                
                # Validate row data
                if sr_no.isdigit():
                    # Process entry transaction
                    entry_date = parse_date(entry_date)
                    qty = int(qty)
                    transactions.append((
                        entry_date,
                        qty,
                        stock_name,
                        float(entry_price)
                    ))
                    
                    # Process exit transaction if exit date exists
                    if len(row) >= 8:
                        exit_date = parse_date(row[7])
                        transactions.append((
                            exit_date,
                            -qty,  # Negative quantity for exits
                            stock_name,
                            float(exit_price)
                        ))

    # Sort transactions by date
    transactions.sort(key=lambda x: x[0])

    # Write output CSV file
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['date', 'qty', 'symbol', 'price'])  # Header
        for transaction in transactions:
            writer.writerow(transaction)

def main():
    """
    Main function for command-line usage.
    
    Example:
        $ python create_txn.py
    """
    # Configuration
    input_file = 'test.tsv'     # Input trading data
    output_file = 'output.csv'  # Standardized output
    
    # Process data
    process_data(input_file, output_file)
    print(f"Output has been written to '{output_file}'")

if __name__ == "__main__":
    main()