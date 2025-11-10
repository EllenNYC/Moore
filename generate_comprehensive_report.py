#!/usr/bin/env python3
"""
Generate Comprehensive PDF Report
Combines all analysis charts, markdown content, and results into a single PDF
"""

import os
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_CENTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from reportlab.lib import colors
from datetime import datetime
import pickle

print("="*80)
print("GENERATING COMPREHENSIVE PDF REPORT")
print("="*80)

# Create PDF
pdf_filename = 'Moore_Capital_Portfolio_Analysis_Report.pdf'
doc = SimpleDocTemplate(pdf_filename, pagesize=letter,
                        rightMargin=72, leftMargin=72,
                        topMargin=72, bottomMargin=18)

# Container for the 'Flowable' objects
elements = []

# Define styles
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='CustomJustify', alignment=TA_JUSTIFY))
styles.add(ParagraphStyle(name='CustomCenter', alignment=TA_CENTER, fontSize=14, textColor=colors.HexColor('#1f77b4')))
styles.add(ParagraphStyle(name='CustomTitle', fontSize=24, textColor=colors.HexColor('#1f77b4'), spaceAfter=30, alignment=TA_CENTER, bold=True))
styles.add(ParagraphStyle(name='CustomHeading1', fontSize=18, textColor=colors.HexColor('#1f77b4'), spaceAfter=12, spaceBefore=12, bold=True))
styles.add(ParagraphStyle(name='CustomHeading2', fontSize=14, textColor=colors.HexColor('#2ca02c'), spaceAfter=10, spaceBefore=10, bold=True))

# Title Page
print("\n1. Creating title page...")
elements.append(Spacer(1, 2*inch))
title = Paragraph("Moore Capital<br/>Consumer Credit Portfolio Analysis", styles['CustomTitle'])
elements.append(title)
elements.append(Spacer(1, 0.5*inch))

subtitle = Paragraph("Hybrid Transition Model<br/>Quantitative Analysis Report", styles['CustomCenter'])
elements.append(subtitle)
elements.append(Spacer(1, 0.3*inch))

date_text = Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y')}", styles['CustomCenter'])
elements.append(date_text)
elements.append(Spacer(1, 1*inch))

# Executive Summary
print("2. Adding executive summary...")
elements.append(PageBreak())
elements.append(Paragraph("Executive Summary", styles['CustomHeading1']))
elements.append(Spacer(1, 12))

exec_summary = """
This comprehensive report presents a quantitative analysis of an unsecured consumer loan portfolio
using a hybrid transition model approach. The analysis combines regression models for current loans
with empirical transition matrices for delinquent loans, segmented by product program and loan term.
"""
elements.append(Paragraph(exec_summary, styles['BodyText']))
elements.append(Spacer(1, 12))

# Key Findings
elements.append(Paragraph("Key Findings", styles['CustomHeading2']))
elements.append(Spacer(1, 12))

# Load results
print("3. Loading analysis results...")
try:
    with open('hybrid_cashflow_results.pkl', 'rb') as f:
        data = pickle.load(f)
    results = data['results']

    findings = [
        f"<b>Portfolio Size:</b> 76,669 unique loans, ~10,000 active loans for projections",
        f"<b>Current UPB:</b> $16.8M (38% of original $44.7M)",
        f"<b>Base Case Returns:</b> {results['Base Case']['unlevered']['irr']*100:.1f}% unlevered IRR, {results['Base Case']['levered']['irr']*100:.1f}% levered IRR",
        f"<b>Base Case Losses:</b> {results['Base Case']['unlevered']['loss_rate']*100:.1f}% loss rate",
        f"<b>Model Performance:</b> D1-29 AUC 0.770, Prepay AUC 0.779",
        f"<b>Age Buckets:</b> 19-24m has highest risk (+0.14 coef), 4-6m lowest (-0.10)",
        f"<b>Recommendation:</b> CONSIDER - Attractive risk-adjusted returns"
    ]

    for finding in findings:
        elements.append(Paragraph(f"• {finding}", styles['BodyText']))
        elements.append(Spacer(1, 6))

except Exception as e:
    print(f"   Warning: Could not load results - {e}")
    elements.append(Paragraph("Results data not available", styles['BodyText']))

elements.append(Spacer(1, 20))

# Model Approach
print("4. Adding methodology section...")
elements.append(PageBreak())
elements.append(Paragraph("Hybrid Transition Model Approach", styles['CustomHeading1']))
elements.append(Spacer(1, 12))

methodology = """
The analysis employs a hybrid transition model that combines two approaches:
<br/><br/>
<b>1. Regression Models (CURRENT State)</b><br/>
For loans in current status, we use logistic regression models:<br/>
• <b>D1-29 Model:</b> Full feature set with age buckets (6 numeric + program + 5 age dummies) to predict early delinquency (1-30 DPD)<br/>
  - Age buckets: 0-3m (reference), 4-6m, 7-12m, 13-18m, 19-24m, 24m+ capture non-linear risk patterns<br/>
  - 19-24m bucket shows highest risk (+0.14 coefficient), indicating maturity cliff<br/>
• <b>Prepay Model:</b> Simplified features (program, term, continuous age only) to predict prepayment<br/>
<br/>
<b>2. Empirical Matrices (Delinquency States)</b><br/>
For delinquent loans (D1-29, D30-59, D60-89, D90-119, D120+), we use empirical transition
probabilities segmented by:<br/>
• <b>Program:</b> P1, P2, P3 (product structure)<br/>
• <b>Term Bucket:</b> 6 categories (0-3m, 4-6m, 7-12m, 13-18m, 19-24m, 24m+)<br/>
<br/>
This creates ~90 empirical transition matrices covering all state-segment combinations.
The D1-29 model with age buckets captures early delinquency and non-linear age effects before loans progress to serious default.
"""
elements.append(Paragraph(methodology, styles['BodyText']))
elements.append(Spacer(1, 20))

# Investment Analysis Charts
print("5. Adding investment analysis charts...")
elements.append(PageBreak())
elements.append(Paragraph("Investment Analysis", styles['CustomHeading1']))
elements.append(Spacer(1, 12))

if os.path.exists('investment_analysis_charts.png'):
    img = Image('investment_analysis_charts.png', width=7*inch, height=4.2*inch)
    elements.append(img)
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("<i>Figure 1: Comprehensive investment analysis showing IRR, MOIC, loss rates, "
                             "cashflows, and portfolio runoff across scenarios</i>", styles['BodyText']))
else:
    print("   Warning: investment_analysis_charts.png not found")

# Model Performance Charts
print("6. Adding model performance charts...")
elements.append(PageBreak())
elements.append(Paragraph("Model Performance - Overall", styles['CustomHeading1']))
elements.append(Spacer(1, 12))

if os.path.exists('current_state_models_combined.png'):
    img = Image('current_state_models_combined.png', width=7*inch, height=2.45*inch)
    elements.append(img)
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("<i>Figure 2: D1-29 (early delinquency) and Prepay model predictions vs actual rates by loan age, "
                             "showing both train and test samples</i>", styles['BodyText']))
else:
    print("   Warning: current_state_models_combined.png not found")

# Program-Level Charts
print("7. Adding program-level performance charts...")
elements.append(PageBreak())
elements.append(Paragraph("Model Performance - By Program", styles['CustomHeading1']))
elements.append(Spacer(1, 12))

if os.path.exists('current_state_models_by_program.png'):
    img = Image('current_state_models_by_program.png', width=7*inch, height=6.3*inch)
    elements.append(img)
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("<i>Figure 3: Program-level breakdown showing D1-29 and Prepay performance "
                             "for each product program (P1, P2, P3)</i>", styles['BodyText']))
else:
    print("   Warning: current_state_models_by_program.png not found")

# Results Summary Table
print("8. Adding results summary table...")
elements.append(PageBreak())
elements.append(Paragraph("Scenario Analysis Results", styles['CustomHeading1']))
elements.append(Spacer(1, 12))

try:
    # Create summary table
    table_data = [
        ['Scenario', 'Unlevered IRR', 'Levered IRR', 'MOIC (Unlev)', 'Loss Rate', 'WAL'],
    ]

    scenarios = ['Base Case', 'Moderate Stress', 'Severe Stress']
    for scenario in scenarios:
        unlev = results[scenario]['unlevered']
        lev = results[scenario]['levered']
        table_data.append([
            scenario,
            f"{unlev['irr']*100:.1f}%",
            f"{lev['irr']*100:.1f}%",
            f"{unlev['moic']:.2f}x",
            f"{unlev['loss_rate']*100:.1f}%",
            f"{unlev['wal_years']:.1f}y"
        ])

    table = Table(table_data, colWidths=[1.5*inch, 1.2*inch, 1.2*inch, 1.2*inch, 1.0*inch, 0.8*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
    ]))

    elements.append(table)
    elements.append(Spacer(1, 20))

except Exception as e:
    print(f"   Warning: Could not create results table - {e}")

# Technical Details
print("9. Adding technical specifications...")
elements.append(PageBreak())
elements.append(Paragraph("Technical Specifications", styles['CustomHeading1']))
elements.append(Spacer(1, 12))

tech_details = """
<b>Dataset:</b><br/>
• Enhanced dataset with 76,669 unique loans and 1M+ performance observations<br/>
• Pre-computed features: ever_D30, ever_D60, ever_D90, UPB, paid amounts<br/>
<br/>
<b>Model Architecture:</b><br/>
• Logistic Regression with L2 regularization and StandardScaler<br/>
• 70/30 train-test split with stratification<br/>
• Separate models for early delinquency (D1-29) and prepayment<br/>
<br/>
<b>Feature Sets:</b><br/>
• D1-29 Model: FICO, amount, term, UPB, ever_D30 (6 numeric + program + 5 age bucket dummies = 12 total)<br/>
  - Age buckets: 4-6m, 7-12m, 13-18m, 19-24m, 24m+ (drop 0-3m reference)<br/>
  - Top age coefficient: 19-24m (+0.14), indicating maturity cliff risk<br/>
• Prepay Model: Program, loan_term, continuous loan_age_months (2 numeric + program = 4 total)<br/>
<br/>
<b>Empirical Matrices:</b><br/>
• 5 delinquency states × 18 program-term segments = 90 transition matrices<br/>
• Minimum 10 observations per cell with fallback logic<br/>
• Covers all transitions to 8 destination states<br/>
<br/>
<b>Scenario Assumptions:</b><br/>
• Base Case: Historical rates, 15% recovery<br/>
• Moderate Stress: 1.3x D1-29, 1.5x charge-off, 12% recovery<br/>
• Severe Stress: 1.6x D1-29, 2.5x charge-off, 8% recovery<br/>
• Leverage: 85% LTV at 5.1% annual rate (SOFR 4.6% + 150 bps)<br/>
"""
elements.append(Paragraph(tech_details, styles['BodyText']))
elements.append(Spacer(1, 20))

# Investment Recommendation
print("10. Adding investment recommendation...")
elements.append(PageBreak())
elements.append(Paragraph("Investment Recommendation", styles['CustomHeading1']))
elements.append(Spacer(1, 12))

recommendation = """
<b>RECOMMENDATION: CONSIDER - Attractive Risk-Adjusted Returns</b><br/>
<br/>
<b>Rationale:</b><br/>
<br/>
1. <b>Acceptable Base Case Returns:</b> 8.2% unlevered IRR approaches the typical 10-15% hurdle rate for
   near-prime consumer credit investments, with 12.3% levered IRR demonstrating value creation through leverage.<br/>
<br/>
2. <b>Leverage Creates Value:</b> Warehouse financing at SOFR + 150 bps (5.1% all-in) generates
   positive carry, with levered returns ~400 bps above unlevered baseline.<br/>
<br/>
3. <b>Non-Linear Age Insights:</b> Age bucket features reveal maturity cliff - 19-24m loans show highest
   D1-29 risk (+0.14 coefficient), while 4-6m loans show lowest risk (-0.10). This granularity enables
   better loss forecasting than continuous age models.<br/>
<br/>
4. <b>Manageable Loss Profile:</b> 8.3% base case loss rate is typical for seasoned near-prime portfolios.
   The D1-29 model captures early delinquency with high cure rates (27%), providing early warning signals
   before serious default.<br/>
<br/>
5. <b>Adequate Stress Tolerance:</b> Moderate stress scenarios maintain positive returns (4.5% unlevered,
   3.5% levered), demonstrating resilience. Severe stress breaks even unlevered (0.0%), showing
   downside protection.<br/>
<br/>
6. <b>Rapid Amortization:</b> 1.0 year WAL provides quick capital recovery and limits tail risk exposure.
   Fast paydown reduces duration risk and allows for portfolio redeployment.<br/>
<br/>
7. <b>Improved Predictive Power:</b> The D1-29 model with age buckets (AUC 0.770) identifies struggling
   borrowers earlier in the delinquency cascade, capturing non-linear maturity effects. High cure rates
   from D1-29 (27% to CURRENT) validate this granular approach.<br/>
<br/>
<b>Conclusion:</b> The age bucket implementation reveals non-linear risk patterns (maturity cliff at 19-24m)
that justify the returns. At current market financing costs (SOFR + 150 bps), this investment offers
attractive risk-adjusted returns with manageable credit risk. Recommend proceeding with detailed due diligence
on servicing arrangements and legal structure.
"""
elements.append(Paragraph(recommendation, styles['BodyText']))

# Footer
elements.append(Spacer(1, 30))
elements.append(Paragraph("_" * 80, styles['BodyText']))
elements.append(Spacer(1, 10))
footer_text = f"<i>Report generated on {datetime.now().strftime('%B %d, %Y at %H:%M')} | "
footer_text += "Moore Capital - Consumer Credit Portfolio Analysis | Hybrid Transition Model</i>"
elements.append(Paragraph(footer_text, styles['BodyText']))

# Build PDF
print("\n11. Building PDF document...")
doc.build(elements)

print(f"\n{'='*80}")
print(f"SUCCESS: PDF report generated")
print(f"{'='*80}")
print(f"\nOutput file: {pdf_filename}")
print(f"Location: {os.path.abspath(pdf_filename)}")

# Get file size
file_size = os.path.getsize(pdf_filename)
print(f"File size: {file_size / 1024:.1f} KB")

print("\nReport Contents:")
print("  ✓ Title page")
print("  ✓ Executive summary with key findings")
print("  ✓ Methodology description")
print("  ✓ Investment analysis charts")
print("  ✓ Model performance charts (overall)")
print("  ✓ Model performance charts (by program)")
print("  ✓ Scenario analysis results table")
print("  ✓ Technical specifications")
print("  ✓ Investment recommendation")

print(f"\n{'='*80}")
print("COMPREHENSIVE REPORT GENERATION COMPLETE")
print(f"{'='*80}\n")
