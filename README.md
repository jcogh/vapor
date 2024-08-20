# VAPOR: Value-at-Risk and Portfolio Optimization Resource

## Description

VAPOR is a Python-based tool for portfolio risk analysis and management. It implements various Value-at-Risk (VaR) calculation methods, performs stress testing, and provides risk mitigation recommendations. The tool generates a PDF report summarizing the analysis results.

## Features

- Multiple VaR calculation methods: Historical, Parametric, and Monte Carlo
- Stress testing with various market scenarios
- Risk appetite assessment
- VaR model backtesting
- Visualization of portfolio returns distribution
- Generation of a comprehensive, PDF report

## Installation

1. Clone this repository:
```
git clone git@github.com:jcogh/vapor.git
cd vapor
```

2. Create and activate a virtual environment:
```
python -m venv vapor_env
source vapor_env/bin/activate  
```
3. Install the required packages:
```
pip install -r requirements.txt
```

## Usage

Run the main script:
```
python vapor_analysis.py
```

This will perform the VaR analysis, stress testing, and generate a PDF report in the 'reports' directory.

## Project Structure
```
vapor/
│
├── vapor_analysis.py
├── requirements.txt
├── README.md
└── reports/
    └── (generated PDF reports will be saved here)
```

## Customization

You can customize the analysis by modifying the following variables in the `main()` function of `vapor_analysis.py`:

- `tickers`: List of stock tickers to analyze
- `weights`: Portfolio weights corresponding to the tickers
- `confidence_level`: Confidence level for VaR calculations
- `risk_appetite`: Risk appetite threshold
- `data_start` and `data_end`: Date range for historical data

## Dependencies

- numpy
- pandas
- yfinance
- matplotlib
- scipy
- seaborn
- reportlab

See `requirements.txt` for specific version information.

## Contributing

Contributions to VAPOR are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
