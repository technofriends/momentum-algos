# Stock Analysis - Momentum Strategy

This project implements and analyzes a momentum-based stock trading strategy.

## Project Structure

```
momentum/
├── src/                    # Source code
│   ├── core/              # Core functionality
│   ├── data/              # Data handling
│   ├── analysis/          # Analysis modules
│   └── visualization/     # Visualization modules
├── data/                  # Data files
├── tests/                # Test files
├── logs/                 # Log files
├── config/              # Configuration files
├── notebooks/           # Jupyter notebooks
├── output/             # Generated output
├── transactions/      # Transaction files
└── requirements.txt   # Project dependencies
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure your settings in `config/config.py`

3. Run portfolio analysis:
```bash
python -m src.analysis.evaluate_portfolio
```

## Features

- Portfolio evaluation and metrics calculation
- Dynamic portfolio analysis
- Portfolio comparison tools
- Data downloading and processing
- Visualization tools

## Dependencies

See `requirements.txt` for a full list of dependencies.
