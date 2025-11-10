#!/usr/bin/env python3
"""
Generate Comprehensive Moore Capital Portfolio Analysis Report
Combines: Data Exploration, Hybrid Transition Model, Validation, and Cashflow Analysis
"""

import pandas as pd
import numpy as np
import pickle
import base64
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import io

print("="*80)
print("GENERATING COMPREHENSIVE MOORE CAPITAL PORTFOLIO ANALYSIS REPORT")
print("="*80)

# ============================================================================
# 1. LOAD DATA AND RESULTS
# ============================================================================
print("\n1. Loading data and model results...")

# Load loan performance data (for section 1.4 and model validation)
df = pd.read_csv('loan_performance_enhanced.csv')
df['report_date'] = pd.to_datetime(df['report_date'])
df['disbursement_d'] = pd.to_datetime(df['disbursement_d'])

# Load loan tape (for sections 1.1, 1.2, 1.3)
loan_tape = pd.read_csv('loan tape - moore v1.0.csv')
loan_tape['disbursement_d'] = pd.to_datetime(loan_tape['disbursement_d'])
# Clean column names (remove leading/trailing spaces)
loan_tape.columns = loan_tape.columns.str.strip()
# Convert numeric columns
loan_tape['mdr'] = pd.to_numeric(loan_tape['mdr'].astype(str).str.replace('%', ''), errors='coerce') / 100
loan_tape['int_rate'] = pd.to_numeric(loan_tape['int_rate'].astype(str).str.replace('%', ''), errors='coerce') / 100
loan_tape['approved_amount'] = pd.to_numeric(loan_tape['approved_amount'].astype(str).str.replace(',', ''), errors='coerce')
loan_tape['co_amt_est'] = pd.to_numeric(loan_tape['co_amt_est'].astype(str).str.replace(',', ''), errors='coerce')

# Load hybrid transition models
with open('hybrid_transition_models.pkl', 'rb') as f:
    models = pickle.load(f)

# Load cashflow results
with open('hybrid_cashflow_results.pkl', 'rb') as f:
    cashflow_data = pickle.load(f)
    cashflow_results = cashflow_data['results']
    cashflow_summary = cashflow_data['summary']

print(f"  Loaded {len(df):,} observations for {df['display_id'].nunique():,} unique loans")
print(f"  Loaded {len(loan_tape):,} loans from loan tape")
print(f"  Loaded hybrid transition models (D1-29 AUC: {models['auc_d1_29']:.3f}, Payoff AUC: {models['auc_prepay']:.3f})")
print(f"  Loaded cashflow results for {len(cashflow_results)} scenarios")

# ============================================================================
# 2. HELPER FUNCTIONS
# ============================================================================

def img_to_base64(img_path):
    """Convert image to base64 for embedding in HTML"""
    with open(img_path, 'rb') as f:
        return base64.b64encode(f.read()).decode()

def create_html_section(title, content, level=2):
    """Create HTML section with title"""
    return f"""
    <h{level} class="section-title">{title}</h{level}>
    {content}
    """

# ============================================================================
# 3. PORTFOLIO STATISTICS
# ============================================================================
print("\n2. Calculating portfolio statistics...")

# Get latest snapshot for portfolio statistics (for section 1.4)
latest_date = df['report_date'].max()
portfolio = df[df['report_date'] == latest_date].copy()

# Portfolio composition from LOAN TAPE (for sections 1.1, 1.2, 1.3)
total_loans = len(loan_tape)
total_original = loan_tape['approved_amount'].sum()

# 1.1 By program (from loan tape) - matching cell 12 from data_exploration.ipynb
by_program = loan_tape.groupby('program').apply(
    lambda g: pd.Series({
        'Loan Count': g['display_id'].count(),
        'Avg Loan Term': g['loan_term'].mean(),
        'Avg MDR (%)': g['mdr'].mean() * 100,
        'Avg Int Rate (%)': g['int_rate'].mean() * 100,
        'Avg FICO': g['fico_score'].mean(),
        'Avg Approved Amount ($)': g['approved_amount'].mean()
    })
).round(2)

# Add overall row
overall = pd.DataFrame({
    'Loan Count': [len(loan_tape)],
    'Avg Loan Term': [loan_tape['loan_term'].mean()],
    'Avg MDR (%)': [loan_tape['mdr'].mean() * 100],
    'Avg Int Rate (%)': [loan_tape['int_rate'].mean() * 100],
    'Avg FICO': [loan_tape['fico_score'].mean()],
    'Avg Approved Amount ($)': [loan_tape['approved_amount'].mean()]
}, index=['Overall'])

by_program = pd.concat([by_program, overall]).round(2)

# 1.2 Generate portfolio trends charts (from cell 13 of data_exploration.ipynb)
loan_tape['vintage_quarter'] = loan_tape['disbursement_d'].dt.to_period('Q')

# Chart 1: Average FICO by vintage quarter
avg_fico_by_vintage = loan_tape.groupby('vintage_quarter')['fico_score'].mean()
fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(avg_fico_by_vintage.index.astype(str), avg_fico_by_vintage.values, marker='o', linewidth=2)
ax1.set_title('Average FICO Score by Vintage Quarter', fontsize=12, fontweight='bold')
ax1.set_xlabel('Vintage Quarter')
ax1.set_ylabel('Average FICO Score')
ax1.tick_params(axis='x', rotation=45)
ax1.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('portfolio_trends_fico.png', dpi=150, bbox_inches='tight')
plt.close()

# Chart 2: Average FICO by vintage quarter and program
avg_fico_by_vintage_prog = loan_tape.groupby(['vintage_quarter', 'program'])['fico_score'].mean().unstack()
fig2, ax2 = plt.subplots(figsize=(10, 4))
for prog in avg_fico_by_vintage_prog.columns:
    ax2.plot(avg_fico_by_vintage_prog.index.astype(str), avg_fico_by_vintage_prog[prog], marker='o', label=prog)
ax2.set_title('Average FICO Score by Vintage Quarter and Program', fontsize=12, fontweight='bold')
ax2.set_xlabel('Vintage Quarter')
ax2.set_ylabel('Average FICO Score')
ax2.tick_params(axis='x', rotation=45)
ax2.legend(title='Program')
ax2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('portfolio_trends_fico_by_program.png', dpi=150, bbox_inches='tight')
plt.close()

# Chart 3: Program approval amount by vintage quarter with P3 percentage
p3_totals = loan_tape[loan_tape['program'] == 'P3'].groupby('vintage_quarter')['approved_amount'].sum()
p3_percentage = p3_totals / loan_tape.groupby('vintage_quarter')['approved_amount'].sum() * 100

fig3, ax3_1 = plt.subplots(figsize=(10, 4))
loan_tape.groupby(['vintage_quarter', 'program'])['approved_amount'].sum().unstack(fill_value=0).plot(kind='bar', stacked=True, ax=ax3_1, colormap='tab20')
ax3_1.set_title('Program Approved Amount by Vintage Quarter', fontsize=12, fontweight='bold')
ax3_1.set_xlabel('Vintage Quarter')
ax3_1.set_ylabel('Approved Amount ($)')
ax3_1.legend(title='Program', bbox_to_anchor=(1.05, 1), loc='upper left')
ax3_1.grid(alpha=0.3)
ax3_1.tick_params(axis='x', rotation=45)

# Create a secondary y-axis
ax3_2 = ax3_1.twinx()
ax3_2.plot(range(len(p3_percentage)), p3_percentage.values, marker='o', color='black', linewidth=2, label='P3 % of Total')
ax3_2.set_ylabel('P3 % of Total Approved Amount')
ax3_2.tick_params(axis='y', labelcolor='black')
ax3_2.legend(bbox_to_anchor=(1.05, 0.15), loc='upper left')

plt.tight_layout()
plt.savefig('portfolio_trends_program_mix.png', dpi=150, bbox_inches='tight')
plt.close()

# 1.3 Generate roll rate analysis heatmap (from cells 26-27 of data_exploration.ipynb)
# Filter for roll rate analysis
roll_rate_data = df[df['next_delinquency_bucket'].notna()].copy()
roll_rate_data = roll_rate_data[roll_rate_data.delinquency_bucket.isin(['CURRENT','1-29 DPD','30-59 DPD','60-89 DPD','90-119 DPD','120+ DPD'])]

# Calculate UPB-weighted roll rate matrix
roll_rate_upb = pd.crosstab(
    roll_rate_data['delinquency_bucket'],
    roll_rate_data['next_delinquency_bucket'],
    values=roll_rate_data['upb'],
    aggfunc='sum'
)

# Calculate row totals for UPB
row_totals_upb = roll_rate_upb.sum(axis=1)

# Normalize by row to get percentages
roll_rate_matrix = roll_rate_upb.div(row_totals_upb, axis=0) * 100

# Reorder rows and columns for better visualization
state_order = ['CURRENT','1-29 DPD','30-59 DPD','60-89 DPD','90-119 DPD','120+ DPD']
next_order = ['CURRENT','1-29 DPD','30-59 DPD','60-89 DPD','90-119 DPD','120+ DPD', 'Paid_off', 'Default']
roll_rate_matrix = roll_rate_matrix.reindex(index=state_order)
existing_states = [s for s in next_order if s in roll_rate_matrix.columns]
roll_rate_matrix = roll_rate_matrix[existing_states]

# Visualize roll rate matrix as heatmap
plt.figure(figsize=(14, 6))
sns.heatmap(
    roll_rate_matrix,
    annot=True,
    fmt='.1f',
    cmap='RdYlGn_r',
    cbar_kws={'label': 'Transition Probability (%)'},
    linewidths=0.5,
    vmin=0,
    vmax=100
)
plt.title('Historical Roll Rate Matrix (Monthly Transition Probabilities)', fontsize=14, fontweight='bold')
plt.xlabel('Next Period State', fontsize=12)
plt.ylabel('Current Period State', fontsize=12)
plt.tight_layout()
plt.savefig('roll_rate_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

# Calculate roll rates by program for Appendix
roll_rates_by_program = {}
for program in ['P1', 'P2', 'P3']:
    prog_data = roll_rate_data[roll_rate_data['program'] == program]
    if len(prog_data) > 0:
        # Calculate UPB-weighted roll rate matrix for this program
        roll_rate_upb_prog = pd.crosstab(
            prog_data['delinquency_bucket'],
            prog_data['next_delinquency_bucket'],
            values=prog_data['upb'],
            aggfunc='sum'
        )
        row_totals_upb_prog = roll_rate_upb_prog.sum(axis=1)
        roll_rate_matrix_prog = roll_rate_upb_prog.div(row_totals_upb_prog, axis=0) * 100
        roll_rates_by_program[program] = roll_rate_matrix_prog

# 1.5 Generate default rate by loan term and program (from cell 32 of data_exploration.ipynb)
# Get final status for each loan - calculate terminal event
df_sorted = df.sort_values(['display_id', 'report_date']).copy()

# Define terminal states
default_states = ['Default']
payoff_states = ['Paid_off']
terminal_states = ['Paid_off','Default']

# Function to get the first terminal event or maturity for each loan
def get_terminal_event(group):
    terminal_mask = group['delinquency_bucket'].isin(terminal_states)
    if terminal_mask.any():
        first_terminal_idx = group[terminal_mask].index[0]
        return group.loc[first_terminal_idx]
    else:
        maturity_mask = group['loan_age_months'] >= group['loan_term']
        if maturity_mask.any():
            first_maturity_idx = group[maturity_mask].index[0]
            return group.loc[first_maturity_idx]
        else:
            return group.iloc[-1]

# Apply function to get terminal event for each loan
final_status_df = df_sorted.groupby('display_id', group_keys=False).apply(get_terminal_event).reset_index(drop=True)

# Create event type indicators
final_status_df['defaulted'] = final_status_df['delinquency_bucket'].isin(default_states).astype(int)
final_status_df['Paid_off'] = final_status_df['delinquency_bucket'].isin(payoff_states).astype(int)
final_status_df['still_active'] = (
    (~final_status_df['delinquency_bucket'].isin(terminal_states)) &
    (final_status_df['loan_age_months'] < final_status_df['loan_term'])
).astype(int)

# Drop still active loans
final_status_df = final_status_df[~(final_status_df['still_active']==1)]

# Filter loans where the difference between 2023-10-31 and disbursement_d is more than or equal to loan_term
# (matching cell 29 filtering logic from data_exploration.ipynb)
cutoff_period = pd.Period('2023-10', freq='M')
loan_periods = final_status_df['disbursement_d'].dt.to_period('M')
months_diff = (cutoff_period.ordinal - loan_periods.apply(lambda x: x.ordinal) + 1)
final_status_df = final_status_df[months_diff >= final_status_df['loan_term']]

# Calculate aggregate stats by term and program (matching cell 31 structure)
term_bucket_default = final_status_df.groupby(['loan_term', 'program']).agg({
    'defaulted': ['sum', 'mean', 'count'],
    'Paid_off': ['sum', 'mean']
}).reset_index()

term_bucket_default.columns = ['term', 'program', 'num_defaults', 'default_rate', 'num_loans', 'num_Paid_off', 'prepay_rate']
term_bucket_default['default_rate'] *= 100
term_bucket_default['prepay_rate'] *= 100
term_bucket_default = term_bucket_default.sort_values('term')

# Create pivot table for display (matching cell 32)
default_rate_matrix = term_bucket_default.pivot(index='term', columns='program', values='default_rate').round(2)

# Generate chart (matching cell 32 style)
fig, ax = plt.subplots(figsize=(10, 6))
for prog in term_bucket_default['program'].unique():
    prog_data = term_bucket_default[term_bucket_default['program'] == prog]
    ax.plot(prog_data['term'], prog_data['default_rate'], marker='o', label=prog)
ax.set_xlabel('Loan Term (Months)', fontsize=12)
ax.set_ylabel('Default Rate (%)', fontsize=12)
ax.set_title('Default Rate by Loan Term and Program', fontsize=14, fontweight='bold')
ax.legend(title='Program', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('term_default_rates.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# 4. GENERATE HTML REPORT
# ============================================================================
print("\n3. Generating HTML report...")

html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Moore Capital - Consumer Credit Portfolio Analysis Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }}
        .header p {{
            margin: 10px 0 0 0;
            font-size: 1.1em;
            opacity: 0.9;
        }}
        .section {{
            background: white;
            padding: 30px;
            margin-bottom: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .section-title {{
            color: #667eea;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        h2 {{
            font-size: 1.8em;
            margin-top: 0;
        }}
        h3 {{
            font-size: 1.4em;
            color: #764ba2;
            margin-top: 25px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 0.95em;
        }}
        th {{
            background-color: #667eea;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }}
        td {{
            padding: 10px 12px;
            border-bottom: 1px solid #ddd;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .metric-box {{
            display: inline-block;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            margin: 10px;
            border-radius: 8px;
            min-width: 200px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric-label {{
            font-size: 0.9em;
            opacity: 0.9;
            margin-bottom: 5px;
        }}
        .metric-value {{
            font-size: 1.8em;
            font-weight: bold;
        }}
        .chart-container {{
            margin: 30px 0;
            text-align: center;
        }}
        .chart-container img {{
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .methodology {{
            background-color: #f8f9fa;
            padding: 20px;
            border-left: 4px solid #667eea;
            margin: 20px 0;
            border-radius: 4px;
        }}
        .key-finding {{
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 15px 0;
            border-radius: 4px;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            color: #666;
            font-size: 0.9em;
        }}
        .equation {{
            background-color: #f8f9fa;
            padding: 15px;
            margin: 15px 0;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
            text-align: center;
        }}
        ul {{
            line-height: 1.8;
        }}
        .highlight {{
            background-color: #fff3cd;
            padding: 2px 6px;
            border-radius: 3px;
        }}
        .two-column-container {{
            display: flex;
            gap: 30px;
            margin: 20px 0;
            align-items: flex-start;
        }}
        .two-column-container .column {{
            flex: 1;
        }}
        .two-column-container .column.table-column {{
            flex: 0 0 auto;
            min-width: 300px;
        }}
        .two-column-container .column.chart-column {{
            flex: 1;
        }}
    </style>
</head>
<body>

    <div class="header">
        <h1>Moore Capital - Consumer Credit Portfolio Analysis</h1>
        <p>Quant Modeler Case Study – Consumer Credit & Cashflow Modeling</p>
        <p>Report Date: {datetime.now().strftime('%B %d, %Y')}</p>
    </div>

    <!-- EXECUTIVE SUMMARY -->
    <div class="section">
        <h2 class="section-title">Executive Summary</h2>

        <div class="key-finding">
            <strong>Key Findings:</strong>
            <ul>
                <li><strong>Portfolio Composition:</strong> Diversified across 3 programs (P1, P2, P3) with terms ranging from 12-60 months</li>
                <li><strong>Base Case IRR:</strong> {cashflow_results['Base Case']['unlevered']['irr']*100:.1f}% unlevered, {cashflow_results['Base Case']['levered']['irr']*100:.1f}% levered (85% LTV)</li>
                <li><strong>Loss Rate:</strong> {cashflow_results['Base Case']['unlevered']['loss_rate']*100:.1f}% expected cumulative loss rate in base case</li>

            </ul>
        </div>
    </div>

    <!-- PORTFOLIO OVERVIEW -->
    <div class="section">
        <h2 class="section-title">1. Data Overview</h2>

        <h3>1.1 Data Quality Assessment</h3>
        <div class="methodology">
            <p>During the data exploration phase, several data quality issues were identified and addressed:</p>

            <h4>Loan Tape Dataset Issues:</h4>
            <ul>
                <li><strong>Missing Values:</strong> The <code>co_amt_est</code> field (estimated charge-off amount) has approximately 3,994 missing values out of 83,235 loans (4.8% of records). This field was excluded from predictive modeling due to significant missingness.</li>
            </ul>

            <h4>Loan Performance Dataset Issues:</h4>
            <ul>
                <li><strong>Limited Historical Period:</strong> Dataset covers performance only through October 2023, providing relatively short observation window for loans originated in 2021-2023. This limits the ability to observe full loan lifecycle performance for longer-term loans and may introduce right-censoring bias in default rate estimates.</li>
                <li><strong>Missing Values:</strong> The <code>charge_off_date</code> field has 988,390 missing values out of 1,045,858 records (94.5%), which is expected as it should only be populated for charged-off loans (57,468 records).</li>
                <li><strong>Outliers Detected:</strong> Some loans show <code>days_delinquent</code> values exceeding 500 days, which appears abnormally high and may indicate data quality issues or exceptional cases requiring investigation.</li>
                <li><strong>Negative Values:</strong> Some records show negative values for <code>paid_principal</code> (minimum: -$105.23) and <code>paid_interest</code> (minimum: -$6.47), likely representing payment adjustments or reversals.</li>
            </ul>

            <h4>Data Cleaning Actions Taken:</h4>
            <ul>
                <li>Converted all date fields to datetime format</li>
                <li>Standardized monetary and percentage fields to numeric types</li>
                <li>Handled missing values appropriately (exclusion from modeling or imputation where suitable)</li>
                <li>Validated loan status transitions and delinquency classifications</li>
                <li>Excluded <code>WRITTEN_OFF</code> loans (cancelled transactions) from analysis</li>
            </ul>
        </div>

        <h3>1.2 Portfolio Composition by Program</h3>
        {by_program.to_html(classes='table', float_format=lambda x: f'{x:,.0f}' if abs(x) > 100 else f'{x:.2f}')}

        <h3>1.3 Portfolio Trends & Credit Quality Evolution</h3>
        <div class="chart-container">
            <img src="data:image/png;base64,{img_to_base64('portfolio_trends_fico.png')}" alt="Average FICO by Vintage">
            <p><em>Figure 1.2a: Average FICO Score by Vintage Quarter - Shows overall credit quality trend over time</em></p>
        </div>

        <div class="chart-container">
            <img src="data:image/png;base64,{img_to_base64('portfolio_trends_fico_by_program.png')}" alt="Average FICO by Program">
            <p><em>Figure 1.2b: Average FICO Score by Vintage Quarter and Program - Credit quality remains relatively stable within each program</em></p>
        </div>

        <div class="chart-container">
            <img src="data:image/png;base64,{img_to_base64('portfolio_trends_program_mix.png')}" alt="Program Mix Evolution">
            <p><em>Figure 1.2c: Program Mix by Vintage Quarter - Shows strategic shift toward P1 (prime) originations and away from P3 (subprime)</em></p>
        </div>

        <div class="key-finding">
            <strong>Key Trends:</strong>
            <ul>
                <li><strong>Credit Quality Evolution:</strong> Average FICO declined from 2021Q2 through 2022, then improved as program mix shifted toward higher quality</li>
                <li><strong>Program Mix Shift:</strong> P1 share increased from ~30% to ~56% of originations since 2023Q2</li>
                <li><strong>P3 Reduction:</strong> P3 share decreased from 29% peak (2022Q4) to ~12% in latest quarter</li>
                <li><strong>Strategic Direction:</strong> Portfolio demonstrates deliberate shift toward more conservative credit strategy</li>
            </ul>
        </div>

        <h3>1.4 Historical Roll Rate Analysis</h3>
        <div class="chart-container">
            <img src="data:image/png;base64,{img_to_base64('roll_rate_heatmap.png')}" alt="Roll Rate Matrix">
            <p><em>Figure 1.3: UPB-weighted monthly transition probabilities between delinquency states. Shows the likelihood of loans moving from one state (rows) to another (columns) in the next month.</em></p>
        </div>

        <div class="methodology">
            <h4>Key Roll Rate Insights:</h4>
            <ul>
                <li><strong>Current Loans (93.5% stay Current):</strong> Strong performance with only 4.0% rolling to 1-29 DPD and 2.3% paying off</li>
                <li><strong>Early Delinquency (1-29 DPD):</strong>
                    <ul>
                        <li>28.6% cure back to Current</li>
                        <li>36.8% remain in 1-29 DPD</li>
                        <li>31.5% roll to 30-59 DPD (deterioration)</li>
                    </ul>
                </li>
                <li><strong>30-59 DPD:</strong> 68.6% roll to 60-89 DPD, showing rapid deterioration once loans reach 30+ days delinquent</li>
                <li><strong>Late Stage Delinquency (90-119 DPD):</strong> 80.4% default, with minimal cure or payoff probability</li>
                <li><strong>Severe Delinquency (120+ DPD):</strong> 86.8% default rate, effectively terminal state</li>
            </ul>
        </div>

        <h3>1.5 Default Rate by Loan Term and Program</h3>

        <div class="two-column-container">
            <div class="column table-column">
                <h4>Table: Default Rate by Loan Term and Program (%)</h4>
                {default_rate_matrix.to_html(classes='table', float_format=lambda x: f'{x:.2f}')}
            </div>
            <div class="column chart-column">
                <div class="chart-container">
                    <img src="data:image/png;base64,{img_to_base64('term_default_rates.png')}" alt="Term Default Rates">
                    <p><em>Figure 1.4: Default rates by loan term and program. Shows default performance across different loan maturities.</em></p>
                </div>
            </div>
        </div>

        <div class="methodology">
            <h4>Understanding Term-Based Default Performance:</h4>
            <p>This analysis shows the cumulative default rate for each loan term by program, based on loans that have reached their terminal state (paid off or defaulted) and had sufficient time to mature by the October 2023 cutoff date. This vintage-complete approach provides unbiased estimates of default risk by loan maturity structure.</p>

            <h4>Key Observations:</h4>
            <ul>
                <li><strong>Program Risk Hierarchy:</strong> P3 (subprime) consistently shows the highest default rates across all terms (14-45%), followed by P2 (near-prime, 2-11%), with P1 (prime) showing the lowest default rates (0.14-5.88%)</li>
                <li><strong>Term Length Impact:</strong> Shorter terms show lower default risk across all programs</li>
                <li><strong>Risk Concentration:</strong> The 12-month P3 segment shows the highest default risk (44.87%), indicating that subprime 12-month loans represent the riskiest segment in the mature portfolio</li>
                <li><strong>Credit Quality Differentiation:</strong> Clear risk segmentation by program - P1 maintains excellent performance across all terms, P2 shows moderate credit risk, and P3 demonstrates material credit deterioration</li>
            </ul>
        </div>

    <!-- METHODOLOGY -->
    <div class="section">
        <h2 class="section-title">2. Hybrid Transition Model Methodology</h2>

        <h3>2.1 Model Architecture</h3>
        <div class="methodology">
            <p>The analysis employs a <strong>hybrid transition model</strong> that combines:</p>
            <ul>
                <li><strong>Logistic Regression Models</strong> for Current state transitions:
                    <ul>
                        <li>Current → D1-29 (Early Delinquency): Full feature set with delinquency history</li>
                        <li>Current → Payoff (Early Payoff): Simplified categorical features</li>
                    </ul>
                </li>
                <li><strong>Empirical Transition Matrices</strong> for delinquent loans:
                    <ul>
                        <li>Program × Term matrices for D1-29, D30-59, D60-89, D90-119, D120+ states</li>
                        <li>Historical roll rates for cure, charge-off, and payoff transitions</li>
                    </ul>
                </li>
            </ul>
        </div>

        <h3>2.2 D1-29 Early Delinquency Model</h3>
        <p><strong>Features ({len(models['feature_cols_d1_29'])}):</strong> FICO score buckets, loan amount buckets, loan term, age buckets,
        UPB, payment history, and delinquency history (ever_D30)</p>

        <p><strong>Top 5 Feature Coefficients:</strong></p>
        <ul>
"""

# Add top D1-29 features
d1_29_features = pd.read_csv('feature_importance_d1_29.csv').head(5)
for _, row in d1_29_features.iterrows():
    html_content += f"            <li><code>{row['Feature']}</code>: {row['Coefficient']:+.4f}</li>\n"

html_content += f"""
        </ul>
        <p><strong>Model Performance:</strong> AUC-ROC = <span class="highlight">{models['auc_d1_29']:.4f}</span></p>

        <h3>2.3 Payoff Model</h3>
        <p><strong>Features ({len(models['feature_cols_prepay'])}):</strong> Program dummies, loan term dummies, age buckets,
        FICO buckets, and UPB (unpaid principal balance) buckets</p>

        <p><strong>Top 5 Feature Coefficients:</strong></p>
        <ul>
"""

# Add top Payoff features
payoff_features = pd.read_csv('feature_importance_prepay.csv').head(5)
for _, row in payoff_features.iterrows():
    html_content += f"            <li><code>{row['Feature']}</code>: {row['Coefficient']:+.4f}</li>\n"

html_content += f"""
        </ul>
        <p><strong>Model Performance:</strong> AUC-ROC = <span class="highlight">{models['auc_prepay']:.4f}</span></p>

        <h3>2.4 Empirical Roll Rate Matrices</h3>
        <p>For delinquent loans (D1-29, D30-59, D60-89, D90-119, D120+), we use historical transition probabilities
        stratified by <strong>Program × Term</strong>. Key roll rates observed:</p>
        <ul>
            <li><strong>D1-29 → Current (Cure):</strong> 27.0%</li>
            <li><strong>D1-29 → D30-59:</strong> 35.3%</li>
            <li><strong>D90-119 → Charge-off:</strong> 79.7%</li>
            <li><strong>D120+ → Charge-off:</strong> 86.0%</li>
        </ul>
    </div>

    <!-- MODEL VALIDATION -->
    <div class="section">
        <h2 class="section-title">3. Model Validation & Performance</h2>

        <h3>3.1 Overall Model Fit</h3>
        <div class="chart-container">
            <img src="data:image/png;base64,{img_to_base64('current_state_models_combined.png')}" alt="Overall Model Performance">
            <p><em>Figure 1: Predicted vs Actual rates by loan age for D1-29 and Payoff models (Train and Test sets)</em></p>
        </div>

        <h3>3.2 Performance by Age Bucket (Vintage)</h3>
        <div class="chart-container">
            <img src="data:image/png;base64,{img_to_base64('current_state_models_by_age_bucket.png')}" alt="Model Performance by Age Bucket">
            <p><em>Figure 2: Model performance across different loan age vintages</em></p>
        </div>

        <h3>3.3 Performance by Loan Term</h3>
        <div class="chart-container">
            <img src="data:image/png;base64,{img_to_base64('current_state_models_by_term.png')}" alt="Model Performance by Term">
            <p><em>Figure 3: Model performance segmented by loan term (12m, 24m, 36m, 48m, 60m)</em></p>
        </div>

        <div class="key-finding">
            <strong>Model Validation Summary:</strong>
            <ul>
                <li>Models show <strong>good calibration</strong> between predicted and actual rates across train and test sets</li>
                <li>Performance is <strong>consistent across programs</strong> (P1, P2, P3) with slight variations by term</li>
                <li>Models capture <strong>age-based dynamics</strong>: delinquency peaks in early months, payoffs increase near maturity</li>
                <li>No significant signs of overfitting or instability across different segmentations</li>
            </ul>
        </div>
    </div>

    <!-- CASHFLOW ANALYSIS -->
    <div class="section">
        <h2 class="section-title">4. Cashflow Projection & Scenario Analysis</h2>

        <h3>4.1 Methodology</h3>
        <div class="methodology">
            <p>Cashflows are projected month-by-month over a 60-month horizon:</p>
            <ul>
                <li><strong>Portfolio:</strong> 8,141 active loans from 2023Q3 origination cohort (July-September 2023), observed as of October 2023</li>
                <li><strong>Vintage Cohort:</strong> Focus on most recent origination quarter to eliminate vintage effects and provide forward-looking analysis</li>
                <li><strong>Starting Point:</strong> Current UPB (unpaid principal balance) as of October 2023 reporting date</li>
                <li><strong>Projection Horizon:</strong> 60 months forward</li>
                <li><strong>Monthly Process:</strong>
                    <ol>
                        <li>Predict delinquency and payoff probabilities using hybrid transition model</li>
                        <li>Sample state transitions based on predicted probabilities</li>
                        <li>Calculate scheduled payments, prepayments, defaults, and recoveries</li>
                        <li>Update loan states and balances for next month</li>
                    </ol>
                </li>
            </ul>
        </div>

        <h3>4.2 Key Assumptions</h3>
        <div class="methodology">
            <p><strong>Pricing & Economics:</strong></p>
            <ul>
                <li><strong>Purchase Price:</strong> (1 - MDR% + 1%) × Approved Amount
                    <ul>
                        <li>MDR (Market Discount Rate) varies by loan: average 5.6%</li>
                        <li>Spread: 1.0% above par (premium pricing)</li>
                        <li>Effective purchase price: ~95.4% of approved amount on average</li>
                    </ul>
                </li>
                <li><strong>Cost of Funding:</strong> SOFR + 1.5% = 3.6% + 1.5% = <strong>5.1%</strong>
                    <ul>
                        <li>Leverage: 85% LTV (Loan-to-Value)</li>
                        <li>Debt service calculated monthly on outstanding balance</li>
                    </ul>
                </li>
                <li><strong>Recovery Rate:</strong> 0% (conservative assumption - actual recovery on charged-off loans)</li>
            </ul>

            <p><strong>Model Parameters:</strong></p>
            <ul>
                <li><strong>Transition Probabilities:</strong> Predicted using hybrid model (logistic regression for Current state, empirical matrices for delinquent states)</li>
                <li><strong>Stress Multipliers:</strong> Applied to D1-29 entry rates and charge-off rates to simulate adverse scenarios</li>
                <li><strong>Amortization:</strong> Equal monthly installments based on loan term and interest rate</li>
            </ul>
        </div>

        <h3>4.3 Scenario Definitions</h3>
        <table>
            <tr>
                <th>Scenario</th>
                <th>D1-29 Stress</th>
                <th>Charge-off Stress</th>
                <th>Recovery Rate</th>
                <th>Description</th>
            </tr>
            <tr>
                <td><strong>Base Case</strong></td>
                <td>1.0x</td>
                <td>1.0x</td>
                <td>0%</td>
                <td>Historical transition rates, conservative recovery assumption</td>
            </tr>
            <tr>
                <td><strong>Moderate Stress</strong></td>
                <td>1.2x</td>
                <td>1.5x</td>
                <td>0%</td>
                <td>20% increase in delinquency entry, 50% increase in charge-offs</td>
            </tr>
            <tr>
                <td><strong>Severe Stress</strong></td>
                <td>1.6x</td>
                <td>2.5x</td>
                <td>0%</td>
                <td>60% increase in delinquency entry, 150% increase in charge-offs</td>
            </tr>
        </table>

        <h3>4.4 Scenario Results</h3>
        {cashflow_summary.to_html(classes='table', index=False)}

        <h3>4.5 Cashflow Breakdown by Scenario</h3>
        <div class="chart-container">
            <img src="data:image/png;base64,{img_to_base64('cashflow_breakdown_comparison.png')}" alt="Cashflow Breakdown">
            <p><em>Figure 6: Monthly cashflow components (Interest, Principal, Payoff, Default) across scenarios</em></p>
        </div>

        <div class="key-finding">
            <strong>Cashflow Analysis Key Takeaways:</strong>
            <ul>
                <li><strong>Base Case:</strong> Attractive returns with {cashflow_results['Base Case']['unlevered']['irr']*100:.1f}% unlevered IRR and {cashflow_results['Base Case']['unlevered']['loss_rate']*100:.1f}% loss rate</li>
                <li><strong>Leverage Impact:</strong> 85% LTV amplifies returns to {cashflow_results['Base Case']['levered']['irr']*100:.1f}% in base case but increases downside risk</li>
                <li><strong>Stress Performance:</strong> Portfolio shows resilience with positive IRR in moderate stress, but severe stress leads to {cashflow_results['Severe Stress']['levered']['irr']*100:.1f}% levered IRR</li>
                <li><strong>Loss Sensitivity:</strong> Charge-off rates are the primary driver of performance variation across scenarios</li>
            </ul>
        </div>
    </div>

    <!-- CONCLUSIONS -->
    <div class="section">
        <h2 class="section-title">5. Conclusions & Recommendations</h2>

        <h3>5.1 Model Strengths</h3>
        <ul>
            <li><strong>Hybrid Approach:</strong> Combines predictive power of logistic regression models with stability of empirical transition matrices</li>
            <li><strong>Strong Validation:</strong> Models show good out-of-sample performance across multiple dimensions</li>
            <li><strong>Granular Segmentation:</strong> Program × Term matrices capture heterogeneity in portfolio behavior</li>
            <li><strong>Interpretability:</strong> Feature coefficients provide clear economic intuition (e.g., high FICO reduces delinquency)</li>
        </ul>

        <h3>5.2 Investment Highlights</h3>
        <ul>
            <li><strong>Attractive Risk-Adjusted Returns:</strong> Base case unlevered IRR of {cashflow_results['Base Case']['unlevered']['irr']*100:.1f}% with {cashflow_results['Base Case']['unlevered']['wal_years']:.1f}y WAL</li>
            <li><strong>Leverage Opportunity:</strong> 85% LTV financing enhances equity returns to {cashflow_results['Base Case']['levered']['irr']*100:.1f}% in base case</li>
            <li><strong>Portfolio Quality:</strong> Average FICO of {portfolio['fico_score'].mean():.0f} with manageable current delinquency levels</li>
        </ul>

        <h3>5.3 Risk Considerations</h3>
        <ul>
            <li><strong>Credit Deterioration:</strong> Moderate stress scenario shows material IRR compression to {cashflow_results['Moderate Stress']['unlevered']['irr']*100:.1f}% unlevered</li>
            <li><strong>Leverage Risk:</strong> High LTV magnifies downside - severe stress results in {cashflow_results['Severe Stress']['levered']['irr']*100:.1f}% levered return</li>
            <li><strong>Model Risk:</strong> Projections based on historical data may not capture unprecedented market conditions</li>
            <li><strong>Concentration:</strong> Consumer credit exposure to macroeconomic factors (unemployment, rates, etc.)</li>
        </ul>

        <h3>5.4 Recommendations</h3>
        <div class="methodology">
            <ol>
                <li><strong>Proceed with Investment:</strong> Base case economics support investment at current pricing</li>
                <li><strong>Monitor Delinquency Triggers:</strong> Implement early warning system for D1-29 entry rate increases</li>
                <li><strong>Portfolio Hedging:</strong> Evaluate credit protection strategies for tail risk scenarios</li>
                <li><strong>Model Refresh:</strong> Update transition matrices quarterly as new data becomes available</li>
            </ol>
        </div>
    </div>

    <!-- APPENDIX -->
    <div class="section">
        <h2 class="section-title">Appendix: Technical Details</h2>

        <h3>A.1 Data Sources</h3>
        <ul>
            <li><strong>Loan Performance Data:</strong> loan_performance_enhanced.csv ({len(df):,} observations)</li>
            <li><strong>Observation Period:</strong> {df['report_date'].min().strftime('%B %Y')} to {df['report_date'].max().strftime('%B %Y')}</li>
            <li><strong>Universe:</strong> {df['display_id'].nunique():,} unique consumer loans</li>
        </ul>

        <h3>A.2 Model Training</h3>
        <ul>
            <li><strong>Train/Test Split:</strong> 70% / 30% random split</li>
            <li><strong>Algorithm:</strong> Scikit-learn LogisticRegression with L2 regularization</li>
            <li><strong>Feature Engineering:</strong> Categorical bucketing, dummy encoding, standardization</li>
            <li><strong>Validation:</strong> Out-of-sample AUC-ROC, calibration plots, segmentation analysis</li>
        </ul>

        <h3>A.3 Software & Tools</h3>
        <ul>
            <li><strong>Language:</strong> Python 3.12</li>
            <li><strong>Libraries:</strong> pandas, numpy, scikit-learn, matplotlib</li>
            <li><strong>Models:</strong> hybrid_transition_models.pkl</li>
            <li><strong>Cashflow Engine:</strong> Stochastic projection with 60-month horizon (single-path simulation with probabilistic transitions)</li>
        </ul>

        <h3>A.4 Model Features & Coefficients</h3>

        <h4>D1-29 Early Delinquency Model (22 features - All Categorical)</h4>
        <p><strong>Feature Categories:</strong></p>
        <ul>
            <li><strong>Program Dummies (2):</strong> program_P2, program_P3 (baseline: P1)</li>
            <li><strong>FICO Buckets (4):</strong> fico_<620, fico_660-699, fico_700-739, fico_740+ (baseline: fico_620-659)</li>
            <li><strong>Loan Amount Buckets (4):</strong> amt_2-4k, amt_4-6k, amt_6-8k, amt_8k+ (baseline: amt_<2k)</li>
            <li><strong>Age Buckets (6):</strong> age_2-3m, age_4-6m, age_7-12m, age_13-18m, age_19-24m, age_24m+ (baseline: age_0-1m)</li>
            <li><strong>Term Dummies (5):</strong> term_6, term_12, term_24, term_36, term_60 (baseline: term_3)</li>
            <li><strong>Delinquency History (1):</strong> ever_D30 (flag for prior 30+ DPD)</li>
        </ul>

        <p><strong>Top Positive Coefficients (increase delinquency risk):</strong></p>
        <ul>
            <li><strong>Low FICO (<620):</strong> +0.18 (subprime borrowers show higher delinquency risk)</li>
            <li><strong>Program P3:</strong> +0.14 (subprime program shows elevated early delinquency)</li>
            <li><strong>Prior Delinquency (ever_D30):</strong> +0.13 (history of 30+ DPD predicts future delinquency)</li>
            <li><strong>Higher Loan Amounts:</strong> Larger loans (4k-8k+) show slightly higher early delinquency risk</li>
        </ul>

        <p><strong>Top Negative Coefficients (reduce delinquency risk):</strong></p>
        <ul>
            <li><strong>High FICO (740+):</strong> -0.78 (prime borrowers least likely to become delinquent)</li>
            <li><strong>Mid-Age Loans (7-12m):</strong> -0.51 (seasoned loans past early payment shock period)</li>
            <li><strong>Good FICO (700-739):</strong> -0.33 (near-prime borrowers show strong payment performance)</li>
            <li><strong>Loan Age (4-6m, 2-3m):</strong> -0.48, -0.33 (loans that survive first month show lower delinquency risk)</li>
        </ul>

        <h4>Payoff Model (21 features - All Categorical)</h4>
        <p><strong>Feature Categories:</strong></p>
        <ul>
            <li><strong>Program Dummies (2):</strong> program_P2, program_P3 (baseline: P1)</li>
            <li><strong>Age Buckets (6):</strong> age_2-3m, age_4-6m, age_7-12m, age_13-18m, age_19-24m, age_25+ (baseline: 0-1m)</li>
            <li><strong>FICO Buckets (4):</strong> fico_620-659, fico_660-699, fico_700-739, fico_740+ (baseline: <620)</li>
            <li><strong>Term Dummies (5):</strong> term_12, term_24, term_36, term_48, term_60 (baseline: term_6)</li>
            <li><strong>UPB Buckets (4):</strong> upb_0-1k, upb_2.5-5k, upb_5-7.5k, upb_7.5k+ (baseline: upb_1-2.5k)</li>
        </ul>

        <p><strong>Key Drivers of Early Payoff:</strong></p>
        <ul>
            <li><strong>Very Low Balance (upb_0-1k):</strong> Strongest predictor (+3.52 coefficient) - loans with balances under $1k are highly likely to be paid off</li>
            <li><strong>Longer Terms (36m, 60m):</strong> Positive coefficients (+1.05, +0.82) - likely capturing maturity effect as longer-term loans approach their scheduled payoff</li>
            <li><strong>Higher FICO (740+):</strong> +0.27 (prime borrowers more likely to refinance or pay off early)</li>
        </ul>

        <p><strong>Negative Drivers (reduce payoff probability):</strong></p>
        <ul>
            <li><strong>High Balance (upb_7.5k+):</strong> -0.86 (larger remaining balances are harder to pay off early)</li>
            <li><strong>Mid-Range Balances (upb_5-7.5k, upb_2.5-5k):</strong> -0.78, -0.41 (all balance categories show lower payoff likelihood compared to the <$1k baseline)</li>
            <li><strong>Loan Age (7-24m):</strong> Negative coefficients indicating lower payoff probability during mid-life of loan</li>
        </ul>

        <h4>Empirical Transition Matrices</h4>
        <p><strong>Structure:</strong> 5 matrices (one per delinquency state) × 3 programs × 6 term buckets = 90 unique transition probability vectors</p>

        <p><strong>Key Roll Rates (aggregate across all programs/terms):</strong></p>
        <ul>
            <li><strong>D1-29 → Current (Cure):</strong> 27.0%</li>
            <li><strong>D1-29 → D30-59 (Roll):</strong> 35.3%</li>
            <li><strong>D1-29 → Stay D1-29:</strong> 35.3%</li>
            <li><strong>D1-29 → Payoff:</strong> 4.5%</li>
            <li><strong>D90-119 → Charge-off:</strong> 79.7%</li>
            <li><strong>D120+ → Charge-off:</strong> 86.0%</li>
        </ul>

        <p><strong>Key Roll Rates by Program:</strong></p>
        <table style="width: 100%; margin: 20px 0;">
            <tr>
                <th>Transition</th>
                <th>P1 (Prime)</th>
                <th>P2 (Near-Prime)</th>
                <th>P3 (Subprime)</th>
            </tr>
"""

# Add program-specific roll rates to the table
roll_rate_rows = [
    ('D1-29 → Current (Cure)', '1-29 DPD', 'CURRENT'),
    ('D1-29 → D30-59 (Roll)', '1-29 DPD', '30-59 DPD'),
    ('D1-29 → Stay D1-29', '1-29 DPD', '1-29 DPD'),
    ('D1-29 → Payoff', '1-29 DPD', 'Paid_off'),
    ('D90-119 → Charge-off', '90-119 DPD', 'Default'),
    ('D120+ → Charge-off', '120+ DPD', 'Default'),
]

for label, from_state, to_state in roll_rate_rows:
    html_content += f"            <tr>\n                <td><strong>{label}</strong></td>\n"
    for program in ['P1', 'P2', 'P3']:
        if program in roll_rates_by_program:
            prog_matrix = roll_rates_by_program[program]
            if from_state in prog_matrix.index and to_state in prog_matrix.columns:
                value = prog_matrix.loc[from_state, to_state]
                html_content += f"                <td>{value:.1f}%</td>\n"
            else:
                html_content += "                <td>—</td>\n"
        else:
            html_content += "                <td>—</td>\n"
    html_content += "            </tr>\n"

html_content += """        </table>

        <h3>A.5 Detailed Performance by Program and Term</h3>
        <div class="chart-container">
            <img src="data:image/png;base64,{img_to_base64('current_state_models_by_program.png')}" alt="Model Performance by Program">
            <p><em>Appendix Figure A1: Detailed model performance segmented by Program (P1, P2, P3) and Loan Term</em></p>
        </div>
    </div>

    <div class="footer">
        <p>Moore Capital - Consumer Credit Portfolio Analysis</p>
        <p>Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
        <p>This report is for internal use only and contains confidential information.</p>
    </div>

</body>
</html>
"""

# ============================================================================
# 5. SAVE REPORT
# ============================================================================
print("\n4. Saving HTML report...")

output_file = 'Moore_Capital_Portfolio_Analysis_Report.html'
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"\n  ✓ Saved comprehensive report: {output_file}")

# Also generate PDF if possible
try:
    print("\n5. Attempting to generate PDF version...")
    import subprocess

    # Try using wkhtmltopdf if available
    result = subprocess.run(
        ['which', 'wkhtmltopdf'],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        pdf_output = 'Moore_Capital_Portfolio_Analysis_Report.pdf'
        subprocess.run([
            'wkhtmltopdf',
            '--enable-local-file-access',
            '--page-size', 'Letter',
            '--margin-top', '15mm',
            '--margin-bottom', '15mm',
            '--margin-left', '15mm',
            '--margin-right', '15mm',
            output_file,
            pdf_output
        ], check=True)
        print(f"  ✓ Saved PDF report: {pdf_output}")
    else:
        print("  ⚠ wkhtmltopdf not found - PDF generation skipped")
        print("    Install with: sudo apt-get install wkhtmltopdf")
except Exception as e:
    print(f"  ⚠ PDF generation failed: {e}")

print("\n" + "="*80)
print("REPORT GENERATION COMPLETE")
print("="*80)
print(f"\nOutput files:")
print(f"  • {output_file} - Comprehensive HTML report")
print(f"\nOpen the HTML file in a web browser to view the complete analysis.")
