import csv
from datetime import datetime

def parse_date(date_str):
    return datetime.strptime(date_str, '%d %B %Y').strftime('%Y-%m-%d')

def process_data(input_file, output_file):
    transactions = []
    
    with open(input_file, 'r') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        next(reader)  # Skip header row
        for row in reader:
            if len(row) >= 7:
                sr_no, stock_name, entry_price, exit_price, qty, pnl, entry_date = row[:7]
                if sr_no.isdigit():
                    entry_date = parse_date(entry_date)
                    qty = int(qty)
                    transactions.append((entry_date, qty, stock_name, float(entry_price)))
                    
                    if len(row) >= 8:
                        exit_date = parse_date(row[7])
                        transactions.append((exit_date, -qty, stock_name, float(exit_price)))

    transactions.sort(key=lambda x: x[0])

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['date', 'qty', 'symbol', 'price'])
        for transaction in transactions:
            writer.writerow(transaction)

# Usage
input_file = 'test.tsv'  # Replace with your input TSV file name
output_file = 'output.csv'
process_data(input_file, output_file)

print(f"Output has been written to '{output_file}'")