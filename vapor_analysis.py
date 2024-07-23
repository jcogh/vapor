import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from io import BytesIO
import os

def get_data(tickers, start="2018-01-01", end="2023-12-31"):
    return yf.download(tickers, start=start, end=end)['Adj Close']

def calculate_historical_var(returns, weights, confidence_level, time_horizon=1):
    portfolio_returns = returns.dot(weights)
    return np.percentile(portfolio_returns, 100 - confidence_level) * np.sqrt(time_horizon)

def calculate_parametric_var(returns, weights, confidence_level, time_horizon=1):
    portfolio_returns = returns.dot(weights)
    mu, sigma = portfolio_returns.mean(), portfolio_returns.std()
    return stats.norm.ppf(1 - confidence_level/100) * sigma * np.sqrt(time_horizon) - mu * time_horizon

def calculate_monte_carlo_var(returns, weights, confidence_level, time_horizon=1, simulations=10000):
    mu, cov_matrix = returns.mean().values, returns.cov().values
    correlated_returns = np.random.multivariate_normal(mu, cov_matrix, simulations)
    portfolio_returns = np.dot(correlated_returns, weights)
    return np.percentile(portfolio_returns, 100 - confidence_level) * np.sqrt(time_horizon)

def stress_test(returns, weights, scenario):
    stressed_returns = returns + pd.Series(scenario, index=returns.columns)
    return stressed_returns.dot(weights).iloc[-1]

def backtest_var(returns, weights, var, confidence_level):
    portfolio_returns = returns.dot(weights)
    violations = (portfolio_returns < -var).sum()
    expected_violations = len(portfolio_returns) * (1 - confidence_level/100)
    return violations, expected_violations

def create_custom_styles():
    styles = getSampleStyleSheet()
    styles['Title'].fontName = 'Helvetica-Bold'
    styles['Title'].fontSize = 24
    styles['Title'].leading = 28
    styles['Title'].alignment = 1
    styles['Title'].spaceAfter = 20

    styles['Heading1'].fontName = 'Helvetica-Bold'
    styles['Heading1'].fontSize = 18
    styles['Heading1'].leading = 22
    styles['Heading1'].spaceBefore = 12
    styles['Heading1'].spaceAfter = 6

    styles['BodyText'].fontName = 'Helvetica'
    styles['BodyText'].fontSize = 10
    styles['BodyText'].leading = 14
    styles['BodyText'].spaceBefore = 3
    styles['BodyText'].spaceAfter = 3
    styles['BodyText'].leftIndent = 20

    styles.add(ParagraphStyle(name='TableHeader',
                              parent=styles['BodyText'],
                              fontName='Helvetica-Bold',
                              fontSize=12,
                              leading=14,
                              alignment=1))
    return styles

def create_table(data, colWidths=None):
    table = Table(data, colWidths=colWidths)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('TOPPADDING', (0, 1), (-1, -1), 6),
        ('BOTTOMPADDING', (0, -1), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    return table

def generate_report(var_results, stressed_returns, risk_appetite, backtest_results, plot_buffer, tickers, weights, data_start, data_end):
    pdf_buffer = BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch, leftMargin=0.5*inch, rightMargin=0.5*inch)
    styles = create_custom_styles()
    elements = []

    elements.append(Paragraph("VAPOR: Value-at-Risk and Portfolio Optimization Report", styles['Title']))
    elements.append(Spacer(1, 0.25*inch))

    elements.append(Paragraph("1. Portfolio Composition", styles['Heading1']))
    data_explanation = f"""
    This report analyzes a portfolio consisting of the following assets and weights:
    """
    elements.append(Paragraph(data_explanation, styles['BodyText']))
    
    portfolio_data = [["Asset", "Weight"]] + [[ticker, f"{weight:.2%}"] for ticker, weight in zip(tickers, weights)]
    elements.append(create_table(portfolio_data, colWidths=[2*inch, 1*inch]))
    
    elements.append(Paragraph(f"Historical data from {data_start} to {data_end} has been used for this analysis.", styles['BodyText']))
    elements.append(Spacer(1, 0.25*inch))

    elements.append(Paragraph("2. Value at Risk (VaR) Analysis", styles['Heading1']))
    elements.append(Paragraph("The table below shows the VaR calculated using different methods:", styles['BodyText']))
    var_data = [["Method", "95% 1-day VaR"]] + [[k, f"{v:.2%}"] for k, v in var_results.items()]
    elements.append(create_table(var_data, colWidths=[2.5*inch, 1.5*inch]))
    elements.append(Spacer(1, 0.25*inch))

    elements.append(Paragraph("3. Stress Test Results", styles['Heading1']))
    elements.append(Paragraph("The following table shows the portfolio returns under various stress scenarios:", styles['BodyText']))
    stress_data = [["Scenario", "Stressed Return"]] + [[k, f"{v:.2%}"] for k, v in stressed_returns.items()]
    elements.append(create_table(stress_data, colWidths=[2.5*inch, 1.5*inch]))
    elements.append(Spacer(1, 0.25*inch))

    elements.append(Paragraph("4. Risk Appetite Assessment", styles['Heading1']))
    elements.append(Paragraph(f"Current Risk Appetite: {risk_appetite:.2%}", styles['BodyText']))
    for method, var in var_results.items():
        elements.append(Paragraph(f"• {method}: {'Within' if var < risk_appetite else 'Exceeds'} risk appetite", styles['BodyText']))
    elements.append(Spacer(1, 0.25*inch))

    elements.append(Paragraph("5. VaR Model Backtesting", styles['Heading1']))
    elements.append(Paragraph(f"Number of VaR violations: {backtest_results['violations']}", styles['BodyText']))
    elements.append(Paragraph(f"Expected number of violations: {backtest_results['expected']:.2f}", styles['BodyText']))
    elements.append(Paragraph(f"Conclusion: {backtest_results['conclusion']}", styles['BodyText']))
    elements.append(Spacer(1, 0.25*inch))

    elements.append(PageBreak())
    elements.append(Paragraph("6. Portfolio Returns Distribution", styles['Heading1']))
    img = Image(plot_buffer)
    img.drawHeight = 4*inch
    img.drawWidth = 7*inch
    elements.append(img)
    elements.append(Spacer(1, 0.25*inch))

    elements.append(Paragraph("7. Summary and Conclusion", styles['Heading1']))
    elements.append(Paragraph("Based on the analysis:", styles['BodyText']))
    
    summary_points = [
        f"1. The portfolio's Value at Risk (VaR) at 95% confidence level ranges from {min(var_results.values()):.2%} to {max(var_results.values()):.2%}, depending on the calculation method.",
        f"2. Stress testing reveals that the portfolio is most vulnerable to a {max(stressed_returns, key=stressed_returns.get)} scenario, potentially losing up to {max(stressed_returns.values()):.2%}.",
        f"3. The VaR model's performance is deemed to be \"{backtest_results['conclusion']}\".",
        f"4. {'The current risk level exceeds the defined risk appetite. Risk mitigation strategies should be considered to align with the risk appetite.' if any(var > risk_appetite for var in var_results.values()) else 'The current risk level is within the defined risk appetite. Continue monitoring the portfolio and market conditions.'}"
    ]
    
    for point in summary_points:
        elements.append(Paragraph(point, styles['BodyText']))

    doc.build(elements)
    return pdf_buffer

def main():
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'BND', 'GLD', 'XOM', 'JPM']
    weights = [0.15, 0.15, 0.15, 0.15, 0.1, 0.1, 0.1, 0.1]
    confidence_level = 95
    risk_appetite = -0.05
    data_start = "2018-01-01"
    data_end = "2023-12-31"

    data = get_data(tickers, start=data_start, end=data_end)
    returns = data.pct_change().dropna()

    historical_var = calculate_historical_var(returns, weights, confidence_level)
    parametric_var = calculate_parametric_var(returns, weights, confidence_level)
    monte_carlo_var = calculate_monte_carlo_var(returns, weights, confidence_level)

    var_results = {
        'Historical VaR': historical_var,
        'Parametric VaR': parametric_var,
        'Monte Carlo VaR': monte_carlo_var
    }

    scenarios = {
        "Market Crash": [-0.1] * len(tickers),
        "Tech Selloff": [-0.15, -0.15, -0.15, -0.15, -0.05, 0, -0.05, -0.05],
        "Energy Crisis": [-0.05, -0.05, -0.05, -0.05, 0, 0.1, 0.15, -0.05],
        "Financial Crisis": [-0.1, -0.1, -0.1, -0.1, 0.05, 0.05, -0.05, -0.2]
    }

    stressed_returns = {name: stress_test(returns, weights, scenario) for name, scenario in scenarios.items()}

    historical_violations, expected_violations = backtest_var(returns, weights, historical_var, confidence_level)
    
    if historical_violations <= expected_violations * 1.2 and historical_violations >= expected_violations * 0.8:
        backtest_conclusion = "VaR model performs adequately"
    elif historical_violations > expected_violations * 1.2:
        backtest_conclusion = "VaR model may be underestimating risk"
    else:
        backtest_conclusion = "VaR model may be overestimating risk"

    backtest_results = {
        'violations': historical_violations,
        'expected': expected_violations,
        'conclusion': backtest_conclusion
    }

    plt.figure(figsize=(10, 6))
    portfolio_returns = returns.dot(weights)
    sns.histplot(portfolio_returns, kde=True, stat="density")
    plt.axvline(historical_var, color='r', linestyle='dashed', linewidth=2, label='Historical VaR')
    plt.axvline(parametric_var, color='g', linestyle='dashed', linewidth=2, label='Parametric VaR')
    plt.axvline(monte_carlo_var, color='b', linestyle='dashed', linewidth=2, label='Monte Carlo VaR')
    plt.title("Portfolio Returns Distribution with VaR Estimates")
    plt.xlabel("Daily Returns")
    plt.ylabel("Density")
    plt.legend()

    plot_buffer = BytesIO()
    plt.savefig(plot_buffer, format='png', dpi=300, bbox_inches='tight')
    plot_buffer.seek(0)

    pdf_buffer = generate_report(var_results, stressed_returns, risk_appetite, backtest_results, plot_buffer,
                                 tickers, weights, data_start, data_end)

    os.makedirs('reports', exist_ok=True)

    with open("reports/vapor_report.pdf", "wb") as f:
        f.write(pdf_buffer.getvalue())

    print("Report generated: reports/vapor_report.pdf")

if __name__ == "__main__":
    main()
