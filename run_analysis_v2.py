#!/usr/bin/env python3
"""
Consumer Credit Portfolio Analysis - Revised with Calibrated Assumptions
Moore Capital Case Study
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("CONSUMER CREDIT PORTFOLIO ANALYSIS - MOORE CAPITAL (CALIBRATED)")
print("="*80)

# Load and clean data
loan_tape = pd.read_csv('loan tape - moore v1.0.csv')
loan_performance = pd.read_csv('loan performance - moore v1.0.csv')

loan_tape.columns = loan_tape.columns.str.strip()
loan_tape['mdr'] = pd.to_numeric(loan_tape['mdr'].str.rstrip('%'), errors='coerce') / 100
loan_tape['int_rate'] = pd.to_numeric(loan_tape['int_rate'].str.rstrip('%'), errors='coerce') / 100
loan_tape['approved_amount'] = pd.to_numeric(loan_tape['approved_amount'].str.replace('$', '').str.replace(',', ''), errors='coerce')

loan_performance['report_date'] = pd.to_datetime(loan_performance['report_date'], errors='coerce')

print(f"\nPortfolio Overview:")
print(f"  Total loans: {len(loan_tape):,}")
print(f"  Total principal: ${loan_tape['approved_amount'].sum():,.0f}")
print(f"  Avg loan size: ${loan_tape['approved_amount'].mean():,.0f}")
print(f"  Avg FICO: {loan_tape['fico_score'].mean():.0f}")
print(f"  Avg interest rate: {loan_tape['int_rate'].mean() * 100:.2f}%")
print(f"  Avg term: {loan_tape['loan_term'].mean():.1f} months")

# Get final outcomes
loan_final = loan_performance.sort_values('report_date').groupby('display_id').last().reset_index()
loan_analysis = loan_tape.merge(loan_final[['display_id', 'loan_status']], on='display_id', how='left')
loan_analysis['defaulted'] = loan_analysis['loan_status'].isin(['CHARGED_OFF', 'WRITTEN_OFF']).astype(int)
loan_analysis['prepaid'] = loan_analysis['loan_status'].isin(['PAID_OFF', 'SATISFIED']).astype(int)

# Calculate observed rates
completed = loan_analysis[~loan_analysis['loan_status'].isin(['CURRENT', 'DELINQUENT', 'GRACE_PERIOD'])]
observed_default_rate = completed['defaulted'].mean()
observed_prepay_rate = completed['prepaid'].mean()

print(f"\nHistorical Performance (Completed Loans):")
print(f"  Cumulative default rate: {observed_default_rate * 100:.2f}%")
print(f"  Cumulative prepayment rate: {observed_prepay_rate * 100:.2f}%")

# ============================================================================
# CALIBRATED CASHFLOW PROJECTION
# ============================================================================

def project_cashflows_calibrated(portfolio_df, cumulative_default_rate, cumulative_prepay_rate,
                                 recovery_rate=0.15, months=60):
    """
    Project cashflows using calibrated historical rates.
    """
    n_loans = len(portfolio_df)
    cashflows = []

    balances = portfolio_df['approved_amount'].values.copy()
    monthly_rates = portfolio_df['int_rate'].values / 12
    terms = portfolio_df['loan_term'].values

    # Calculate monthly payment (amortizing)
    monthly_payments = np.zeros(n_loans)
    for i in range(n_loans):
        if monthly_rates[i] > 0 and terms[i] > 0:
            r = monthly_rates[i]
            n = terms[i]
            monthly_payments[i] = balances[i] * (r * (1 + r)**n) / ((1 + r)**n - 1)
        else:
            monthly_payments[i] = balances[i] / max(terms[i], 1)

    # Convert cumulative to monthly conditional rates
    # Assume most defaults/prepays happen in first 24 months
    avg_life_months = 24
    monthly_default_rate = 1 - (1 - cumulative_default_rate) ** (1/avg_life_months)
    monthly_prepay_rate = 1 - (1 - cumulative_prepay_rate) ** (1/avg_life_months)

    loan_status = np.zeros(n_loans)  # 0=active, 1=defaulted, 2=prepaid

    for month in range(months):
        active_mask = (loan_status == 0) & (balances > 0.01)

        if active_mask.sum() == 0:
            break

        # Interest
        interest = np.where(active_mask, balances * monthly_rates, 0)

        # Scheduled principal
        scheduled_principal = np.where(active_mask, np.minimum(monthly_payments - interest, balances), 0)

        # Defaults (apply conditional monthly rate)
        default_mask = active_mask & (np.random.random(n_loans) < monthly_default_rate)
        default_amount = np.where(default_mask, balances, 0)
        recovery_amount = default_amount * recovery_rate

        # Prepayments
        prepay_mask = active_mask & ~default_mask & (np.random.random(n_loans) < monthly_prepay_rate)
        prepay_amount = np.where(prepay_mask, balances, 0)

        # Update status
        loan_status = np.where(default_mask, 1, loan_status)
        loan_status = np.where(prepay_mask, 2, loan_status)

        # Total principal
        total_principal = scheduled_principal + prepay_amount + recovery_amount

        # Update balances
        balances = np.where(default_mask | prepay_mask, 0,
                          np.maximum(balances - scheduled_principal, 0))

        cashflows.append({
            'month': month + 1,
            'interest': interest.sum(),
            'scheduled_principal': scheduled_principal.sum(),
            'prepayments': prepay_amount.sum(),
            'defaults': default_amount.sum(),
            'recoveries': recovery_amount.sum(),
            'total_inflow': interest.sum() + total_principal.sum(),
            'net_loss': default_amount.sum() - recovery_amount.sum(),
            'ending_balance': balances.sum(),
            'active_loans': active_mask.sum()
        })

    return pd.DataFrame(cashflows)

# ============================================================================
# RETURN CALCULATIONS
# ============================================================================

def calculate_irr_newton(cashflows, initial_investment):
    """Calculate IRR using Newton's method."""
    all_cf = [-initial_investment] + list(cashflows)

    def npv(rate):
        return sum([cf / (1 + rate) ** i for i, cf in enumerate(all_cf)])

    try:
        irr_monthly = optimize.newton(npv, 0.001, maxiter=100, tol=1e-6)
        irr_annual = (1 + irr_monthly) ** 12 - 1
        return irr_annual
    except:
        # Fallback: simple return
        total_return = sum(cashflows)
        simple_return = (total_return / initial_investment) - 1
        return simple_return

def calculate_returns(cashflows, initial_investment, leverage_ratio=0.0, cost_of_debt=0.0):
    """Calculate returns with leverage."""

    portfolio_value = initial_investment
    debt_amount = portfolio_value * leverage_ratio
    equity_amount = portfolio_value * (1 - leverage_ratio)

    monthly_debt_rate = cost_of_debt / 12

    equity_cashflows = []
    outstanding_debt = debt_amount

    for _, row in cashflows.iterrows():
        # Interest expense on outstanding debt
        interest_expense = outstanding_debt * monthly_debt_rate

        # Principal collections
        principal_collected = row['scheduled_principal'] + row['prepayments'] + row['recoveries']

        # Pay down debt
        debt_paydown = min(principal_collected, outstanding_debt)
        outstanding_debt -= debt_paydown

        # Equity cashflow = interest earned - interest paid - losses + excess principal
        eq_cf = row['interest'] - interest_expense - row['net_loss'] + (principal_collected - debt_paydown)
        equity_cashflows.append(eq_cf)

    # Calculate metrics
    irr = calculate_irr_newton(equity_cashflows, equity_amount)
    total_returned = sum(equity_cashflows)
    moic = total_returned / equity_amount if equity_amount > 0 else 0

    # WAL
    if total_returned > 0:
        wal = sum([equity_cashflows[i] * (i + 1) for i in range(len(equity_cashflows))]) / total_returned / 12
    else:
        wal = 0

    loss_rate = cashflows['net_loss'].sum() / portfolio_value

    return {
        'portfolio_value': portfolio_value,
        'debt': debt_amount,
        'equity': equity_amount,
        'leverage': f"{leverage_ratio*100:.0f}%",
        'irr': irr,
        'moic': moic,
        'wal_years': wal,
        'total_interest': cashflows['interest'].sum(),
        'total_losses': cashflows['net_loss'].sum(),
        'loss_rate': loss_rate,
        'total_returned': total_returned
    }

# ============================================================================
# RUN SCENARIOS
# ============================================================================

# Use full portfolio or sample
sample_size = min(10000, len(loan_tape))
np.random.seed(42)
portfolio_sample = loan_tape.dropna(subset=['approved_amount', 'int_rate', 'loan_term']).sample(n=sample_size, random_state=42)

initial_value = portfolio_sample['approved_amount'].sum()

print(f"\n" + "="*80)
print("SCENARIO ANALYSIS")
print("="*80)
print(f"\nSample portfolio: {sample_size:,} loans, ${initial_value:,.0f}")

# Define scenarios
scenarios = {
    'Base Case': {
        'default_rate': observed_default_rate,  # ~20%
        'prepay_rate': observed_prepay_rate,    # ~67%
        'recovery': 0.15
    },
    'Moderate Stress': {
        'default_rate': observed_default_rate * 1.3,  # +30%
        'prepay_rate': observed_prepay_rate * 0.85,   # -15%
        'recovery': 0.12
    },
    'Severe Stress': {
        'default_rate': min(observed_default_rate * 1.8, 0.45),  # +80%, capped at 45%
        'prepay_rate': observed_prepay_rate * 0.65,               # -35%
        'recovery': 0.08
    }
}

results = {}
cashflow_results = {}

for scenario_name, params in scenarios.items():
    print(f"\n{scenario_name}:")
    print(f"  Cumulative default rate: {params['default_rate']*100:.1f}%")
    print(f"  Cumulative prepay rate: {params['prepay_rate']*100:.1f}%")
    print(f"  Recovery rate: {params['recovery']*100:.0f}%")

    np.random.seed(42)
    cf = project_cashflows_calibrated(
        portfolio_sample,
        cumulative_default_rate=params['default_rate'],
        cumulative_prepay_rate=params['prepay_rate'],
        recovery_rate=params['recovery']
    )

    cashflow_results[scenario_name] = cf

    # Unlevered
    unlev = calculate_returns(cf, initial_value, 0.0, 0.0)

    # Levered (85% LTV at 6.5%)
    lev = calculate_returns(cf, initial_value, 0.85, 0.065)

    results[scenario_name] = {'unlevered': unlev, 'levered': lev}

    print(f"\n  Unlevered:")
    print(f"    IRR: {unlev['irr']*100:.2f}%")
    print(f"    MOIC: {unlev['moic']:.2f}x")
    print(f"    WAL: {unlev['wal_years']:.2f} years")
    print(f"    Loss rate: {unlev['loss_rate']*100:.2f}%")

    print(f"  Levered (85% LTV, 6.5% debt):")
    print(f"    Equity: ${lev['equity']:,.0f}")
    print(f"    IRR: {lev['irr']*100:.2f}%")
    print(f"    MOIC: {lev['moic']:.2f}x")

# ============================================================================
# SUMMARY TABLE
# ============================================================================

print("\n" + "="*80)
print("SUMMARY TABLE")
print("="*80)

summary_data = []
for scenario_name in ['Base Case', 'Moderate Stress', 'Severe Stress']:
    unlev = results[scenario_name]['unlevered']
    lev = results[scenario_name]['levered']

    summary_data.append({
        'Scenario': scenario_name,
        'Unlevered IRR': f"{unlev['irr']*100:.1f}%",
        'Unlevered MOIC': f"{unlev['moic']:.2f}x",
        'Levered IRR': f"{lev['irr']*100:.1f}%",
        'Levered MOIC': f"{lev['moic']:.2f}x",
        'Loss Rate': f"{unlev['loss_rate']*100:.1f}%",
        'WAL': f"{unlev['wal_years']:.1f}y"
    })

summary_df = pd.DataFrame(summary_data)
print("\n" + summary_df.to_string(index=False))

# Save results
import pickle
with open('analysis_results_v2.pkl', 'wb') as f:
    pickle.dump({
        'results': results,
        'cashflows': cashflow_results,
        'summary': summary_df,
        'portfolio_stats': {
            'n_loans': sample_size,
            'total_value': initial_value,
            'avg_loan': portfolio_sample['approved_amount'].mean(),
            'avg_fico': portfolio_sample['fico_score'].mean(),
            'avg_rate': portfolio_sample['int_rate'].mean(),
            'avg_term': portfolio_sample['loan_term'].mean()
        }
    }, f)

print("\n" + "="*80)
print("ANALYSIS COMPLETE - Results saved to analysis_results_v2.pkl")
print("="*80)
