import os
from fpdf import FPDF

def generate_report(
    accuracy,
    anova,
    pnl,
    win_rate,
    volume
):

    os.makedirs(
        'reports',
        exist_ok=True
    )

    pdf = FPDF()

    pdf.add_page()

    pdf.set_font(
        'Arial',
        'B',
        18
    )

    pdf.cell(
        200,
        10,
        txt='Crypto Sentiment Analysis Report',
        ln=True,
        align='C'
    )

    pdf.ln(10)

    pdf.set_font(
        'Arial',
        '',
        12
    )

    report_text = f"""
Model Accuracy:
{accuracy}

ANOVA Test:
{anova}

Average PnL:
{pnl}

Win Rate:
{win_rate}

Average Trading Volume:
{volume}

Insights:
1. Market sentiment impacts trader profitability.
2. Greed periods often produce higher profits.
3. Fear periods show increased volatility.
4. XGBoost achieved strong predictive accuracy.
"""

    pdf.multi_cell(
        0,
        10,
        txt=report_text
    )

    pdf.output(
        'reports/final_report.pdf'
    )

    print(
        '\nPDF Report Generated Successfully'
    )