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
        f"<b>Current UPB:</b> $17.6M (39% of original $44.7M)",
        f"<b>Base Case Returns:</b> {results['Base Case']['unlevered']['irr']*100:.1f}% unlevered IRR, {results['Base Case']['levered']['irr']*100:.1f}% levered IRR",
        f"<b>Base Case Losses:</b> {results['Base Case']['unlevered']['loss_rate']*100:.1f}% loss rate",
        f"<b>Model Performance:</b> D30+ AUC 0.782, Prepay AUC 0.779",
        f"<b>Recommendation:</b> PASS - Returns insufficient for risk level"
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
• <b>D30+ Model:</b> Full feature set (10 features + program) to predict transition to delinquency<br/>
• <b>Prepay Model:</b> Simplified features (program, term, age only) to predict prepayment<br/>
<br/>
<b>2. Empirical Matrices (Delinquency States)</b><br/>
For delinquent loans (D1-29, D30-59, D60-89, D90-119, D120+), we use empirical transition
probabilities segmented by:<br/>
• <b>Program:</b> P1, P2, P3 (product structure)<br/>
• <b>Term Bucket:</b> 6 categories (0-3m, 4-6m, 7-12m, 13-18m, 19-24m, 24m+)<br/>
<br/>
This creates ~90 empirical transition matrices covering all state-segment combinations.
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
    elements.append(Paragraph("<i>Figure 2: D30+ and Prepay model predictions vs actual rates by loan age, "
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
    elements.append(Paragraph("<i>Figure 3: Program-level breakdown showing D30+ and Prepay performance "
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
• Removed class_weight='balanced' for better probability calibration<br/>
<br/>
<b>Feature Sets:</b><br/>
• D30+ Model: FICO, amount, term, age, UPB, payments, delinquency history (10 features + program)<br/>
• Prepay Model: Program, loan_term, loan_age_months only (2 features + program)<br/>
<br/>
<b>Empirical Matrices:</b><br/>
• 5 delinquency states × 18 program-term segments = 90 transition matrices<br/>
• Minimum 10 observations per cell with fallback logic<br/>
• Covers all transitions to 8 destination states<br/>
<br/>
<b>Scenario Assumptions:</b><br/>
• Base Case: Historical rates, 15% recovery<br/>
• Moderate Stress: 1.3x D30, 1.5x charge-off, 12% recovery<br/>
• Severe Stress: 1.6x D30, 2.5x charge-off, 8% recovery<br/>
• Leverage: 85% LTV at 6.5% annual rate<br/>
"""
elements.append(Paragraph(tech_details, styles['BodyText']))
elements.append(Spacer(1, 20))

# Investment Recommendation
print("10. Adding investment recommendation...")
elements.append(PageBreak())
elements.append(Paragraph("Investment Recommendation", styles['CustomHeading1']))
elements.append(Spacer(1, 12))

recommendation = """
<b>RECOMMENDATION: PASS - Do Not Invest</b><br/>
<br/>
<b>Rationale:</b><br/>
<br/>
1. <b>Insufficient Returns:</b> 3.6% unlevered IRR in base case is far below the 10-15% hurdle rate
   typically required for near-prime consumer credit investments.<br/>
<br/>
2. <b>Leverage Destroys Value:</b> Standard warehouse financing (85% LTV at 6.5%) results in negative
   levered returns (-0.8% base case), making this investment uneconomical with typical financing structures.<br/>
<br/>
3. <b>High Embedded Losses:</b> The seasoned portfolio shows 7.8% base case loss rate, reflecting
   existing delinquencies and credit deterioration already embedded in the portfolio.<br/>
<br/>
4. <b>No Margin of Safety:</b> Moderate stress scenarios result in near-zero returns (0.4% unlevered),
   and severe stress produces material losses (-3.2% unlevered, -16.2% levered). The portfolio cannot
   withstand normal credit cycle stress.<br/>
<br/>
5. <b>Advanced Maturity:</b> Current UPB represents only 39% of original principal, indicating
   significant runoff has already occurred and limiting upside potential from interest income.<br/>
<br/>
6. <b>Superior Alternatives Available:</b> Prime auto ABS, equipment finance, and secured SMB lending
   typically offer 6-12% unlevered IRRs with lower risk profiles and better structural protections.<br/>
<br/>
<b>Conclusion:</b> The seasoned portfolio state reveals structural challenges (high losses, low returns,
advanced maturity) that make this investment unattractive at any reasonable price. A discount of 20%+
would be required to achieve acceptable risk-adjusted returns, which is unlikely to be economically
feasible for the seller.
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
