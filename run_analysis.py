#!/usr/bin/env python3
"""
Consumer Credit Portfolio Analysis
Moore Capital Case Study
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from scipy import optimize
import warnings
warnings.filterwarnings('ignore')

# Set styling
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')

print("="*80)
print("CONSUMER CREDIT PORTFOLIO ANALYSIS - MOORE CAPITAL CASE STUDY")
print("="*80)

# ============================================================================
# 1. LOAD AND CLEAN DATA
# ============================================================================
print("\n1. Loading and cleaning data...")

loan_tape = pd.read_csv('loan tape - moore v1.0.csv')
loan_performance = pd.read_csv('loan performance - moore v1.0.csv')

# Clean loan tape
loan_tape.columns = loan_tape.columns.str.strip()
loan_tape['mdr'] = pd.to_numeric(loan_tape['mdr'].str.rstrip('%'), errors='coerce') / 100
loan_tape['int_rate'] = pd.to_numeric(loan_tape['int_rate'].str.rstrip('%'), errors='coerce') / 100
loan_tape['approved_amount'] = pd.to_numeric(loan_tape['approved_amount'].str.replace('$', '').str.replace(',', ''), errors='coerce')
loan_tape['co_amt_est'] = pd.to_numeric(loan_tape['co_amt_est'], errors='coerce')
loan_tape['disbursement_date'] = pd.to_datetime(loan_tape['disbursement_d'], errors='coerce')

# Clean performance data
loan_performance['report_date'] = pd.to_datetime(loan_performance['report_date'], errors='coerce')
loan_performance['charge_off_date'] = pd.to_datetime(loan_performance['charge_off_date'], errors='coerce')

print(f"  Loaded {len(loan_tape):,} loans")
print(f"  Total portfolio value: ${loan_tape['approved_amount'].sum():,.0f}")

# Get final status
loan_final_status = loan_performance.sort_values('report_date').groupby('display_id').last().reset_index()
loan_analysis = loan_tape.merge(loan_final_status[['display_id', 'loan_status', 'co_amt']], on='display_id', how='left')

# Create outcome flags
loan_analysis['defaulted'] = loan_analysis['loan_status'].isin(['CHARGED_OFF', 'WRITTEN_OFF']).astype(int)
loan_analysis['prepaid'] = loan_analysis['loan_status'].isin(['PAID_OFF', 'SATISFIED']).astype(int)

print(f"  Default rate: {loan_analysis['defaulted'].mean() * 100:.2f}%")
print(f"  Prepayment rate: {loan_analysis['prepaid'].mean() * 100:.2f}%")

# ============================================================================
# 2. BUILD DEFAULT MODEL
# ============================================================================
print("\n2. Building default probability model...")

feature_cols = ['fico_score', 'approved_amount', 'loan_term', 'int_rate', 'mdr']
modeling_data = loan_analysis[loan_analysis['loan_status'] != 'CURRENT'].copy()
modeling_data_clean = modeling_data[feature_cols + ['defaulted']].dropna()

X = modeling_data_clean[feature_cols]
y = modeling_data_clean['defaulted']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

default_model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
default_model.fit(X_train_scaled, y_train)

test_auc = roc_auc_score(y_test, default_model.predict_proba(X_test_scaled)[:, 1])
print(f"  Default model AUC: {test_auc:.4f}")

# ============================================================================
# 3. BUILD PREPAYMENT MODEL
# ============================================================================
print("\n3. Building prepayment probability model...")

prepay_modeling_data = modeling_data[modeling_data['defaulted'] == 0].copy()
prepay_data_clean = prepay_modeling_data[feature_cols + ['prepaid']].dropna()

X_prepay = prepay_data_clean[feature_cols]
y_prepay = prepay_data_clean['prepaid']

X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_prepay, y_prepay, test_size=0.3, random_state=42, stratify=y_prepay)

scaler_prepay = StandardScaler()
X_train_p_scaled = scaler_prepay.fit_transform(X_train_p)
X_test_p_scaled = scaler_prepay.transform(X_test_p)

prepay_model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
prepay_model.fit(X_train_p_scaled, y_train_p)

test_auc_p = roc_auc_score(y_test_p, prepay_model.predict_proba(X_test_p_scaled)[:, 1])
print(f"  Prepayment model AUC: {test_auc_p:.4f}")

# ============================================================================
# 4. CASHFLOW PROJECTION FUNCTION
# ============================================================================

def project_loan_cashflows(loan_data, default_proba, prepay_proba,
                          default_multiplier=1.0, prepay_multiplier=1.0,
                          recovery_rate=0.15, months=60):
    """Project monthly cashflows for a loan portfolio."""

    n_loans = len(loan_data)
    cashflows = []

    balances = loan_data['approved_amount'].values.copy()
    monthly_rates = loan_data['int_rate'].values / 12
    terms = loan_data['loan_term'].values

    # Calculate monthly payment
    monthly_payments = np.zeros(n_loans)
    for i in range(n_loans):
        if monthly_rates[i] > 0 and terms[i] > 0:
            r = monthly_rates[i]
            n = terms[i]
            monthly_payments[i] = balances[i] * (r * (1 + r)**n) / ((1 + r)**n - 1)
        else:
            monthly_payments[i] = balances[i] / max(terms[i], 1)

    # Adjust probabilities
    adj_default_proba = np.clip(default_proba * default_multiplier, 0, 1)
    adj_prepay_proba = np.clip(prepay_proba * prepay_multiplier, 0, 1)

    # Monthly rates
    monthly_default_rate = 1 - (1 - adj_default_proba) ** (1/12)
    monthly_prepay_rate = 1 - (1 - adj_prepay_proba) ** (1/12)

    loan_status = np.zeros(n_loans)

    for month in range(months):
        active_mask = (loan_status == 0) & (balances > 0.01)

        if active_mask.sum() == 0:
            break

        interest = np.where(active_mask, balances * monthly_rates, 0)
        scheduled_principal = np.where(active_mask, np.minimum(monthly_payments - interest, balances), 0)

        # Defaults
        default_this_month = active_mask & (np.random.random(n_loans) < monthly_default_rate)
        default_amount = np.where(default_this_month, balances, 0)
        recovery_amount = default_amount * recovery_rate

        # Prepayments
        prepay_this_month = active_mask & ~default_this_month & (np.random.random(n_loans) < monthly_prepay_rate)
        prepay_amount = np.where(prepay_this_month, balances, 0)

        loan_status = np.where(default_this_month, 1, loan_status)
        loan_status = np.where(prepay_this_month, 2, loan_status)

        total_principal = scheduled_principal + prepay_amount + recovery_amount

        balances = np.where(default_this_month | prepay_this_month, 0,
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
# 5. CALCULATE RETURNS
# ============================================================================

def calculate_irr(cashflows, initial_investment):
    """Calculate IRR using Newton's method."""
    all_cf = [-initial_investment] + list(cashflows)

    def npv(rate):
        return sum([cf / (1 + rate) ** i for i, cf in enumerate(all_cf)])

    try:
        irr_monthly = optimize.newton(npv, 0.01, maxiter=100)
        return (1 + irr_monthly) ** 12 - 1
    except:
        return np.nan

def calculate_returns(cashflows, initial_investment, leverage_ratio=0.0, cost_of_debt=0.0):
    """Calculate investment returns."""

    portfolio_value = initial_investment / (1 - leverage_ratio) if leverage_ratio < 1 else initial_investment
    debt_amount = portfolio_value * leverage_ratio
    equity_amount = portfolio_value - debt_amount

    monthly_debt_cost = cost_of_debt / 12

    equity_cf = []
    outstanding_debt = debt_amount

    for idx, row in cashflows.iterrows():
        interest_expense = outstanding_debt * monthly_debt_cost
        principal_collections = row['scheduled_principal'] + row['prepayments'] + row['recoveries']

        debt_paydown = min(principal_collections, outstanding_debt)
        outstanding_debt -= debt_paydown

        equity_cashflow = row['interest'] - interest_expense - row['net_loss'] + (principal_collections - debt_paydown)
        equity_cf.append(equity_cashflow)

    irr_annual = calculate_irr(equity_cf, equity_amount)

    total_returned = sum(equity_cf)
    moic = total_returned / equity_amount if equity_amount > 0 else 0

    if total_returned > 0:
        wal = sum([equity_cf[i] * (i + 1) for i in range(len(equity_cf))]) / total_returned / 12
    else:
        wal = 0

    total_losses = cashflows['net_loss'].sum()
    loss_rate = total_losses / portfolio_value if portfolio_value > 0 else 0

    return {
        'portfolio_value': portfolio_value,
        'debt_amount': debt_amount,
        'equity_amount': equity_amount,
        'leverage_ratio': leverage_ratio,
        'irr_annual': irr_annual,
        'moic': moic,
        'wal_years': wal,
        'total_interest': cashflows['interest'].sum(),
        'total_losses': total_losses,
        'loss_rate': loss_rate,
        'total_returned': total_returned
    }

# ============================================================================
# 6. RUN SCENARIOS
# ============================================================================
print("\n4. Preparing portfolio for cashflow projections...")

portfolio_data = loan_tape[['approved_amount', 'loan_term', 'int_rate', 'mdr', 'fico_score']].dropna()
X_portfolio = portfolio_data[feature_cols]
X_portfolio_scaled = scaler.transform(X_portfolio)
X_portfolio_scaled_prepay = scaler_prepay.transform(X_portfolio)

default_probabilities = default_model.predict_proba(X_portfolio_scaled)[:, 1]
prepay_probabilities = prepay_model.predict_proba(X_portfolio_scaled_prepay)[:, 1]

print(f"  Portfolio size: {len(portfolio_data):,} loans")
print(f"  Average default probability: {default_probabilities.mean() * 100:.2f}%")
print(f"  Average prepayment probability: {prepay_probabilities.mean() * 100:.2f}%")

# Sample for faster computation
sample_size = min(10000, len(portfolio_data))
np.random.seed(42)
sample_idx = np.random.choice(len(portfolio_data), sample_size, replace=False)
portfolio_sample = portfolio_data.iloc[sample_idx].reset_index(drop=True)
default_proba_sample = default_probabilities[sample_idx]
prepay_proba_sample = prepay_probabilities[sample_idx]

print(f"\n5. Running scenario analysis (using {sample_size:,} loan sample)...")

# Base Case
print("\n  Base Case...")
np.random.seed(42)
cf_base = project_loan_cashflows(portfolio_sample, default_proba_sample, prepay_proba_sample,
                                  default_multiplier=1.0, prepay_multiplier=1.0, recovery_rate=0.15)

# Moderate Stress
print("  Moderate Stress...")
np.random.seed(42)
cf_moderate = project_loan_cashflows(portfolio_sample, default_proba_sample, prepay_proba_sample,
                                      default_multiplier=1.5, prepay_multiplier=0.8, recovery_rate=0.12)

# Severe Stress
print("  Severe Stress...")
np.random.seed(42)
cf_severe = project_loan_cashflows(portfolio_sample, default_proba_sample, prepay_proba_sample,
                                    default_multiplier=2.5, prepay_multiplier=0.5, recovery_rate=0.08)

# ============================================================================
# 7. CALCULATE RETURNS
# ============================================================================
print("\n6. Calculating returns...")

initial_portfolio_value = portfolio_sample['approved_amount'].sum()

# Unlevered
returns_base_unlev = calculate_returns(cf_base, initial_portfolio_value, 0.0)
returns_mod_unlev = calculate_returns(cf_moderate, initial_portfolio_value, 0.0)
returns_sev_unlev = calculate_returns(cf_severe, initial_portfolio_value, 0.0)

# Levered 85% LTV at 6.5%
returns_base_lev = calculate_returns(cf_base, initial_portfolio_value, 0.85, 0.065)
returns_mod_lev = calculate_returns(cf_moderate, initial_portfolio_value, 0.85, 0.065)
returns_sev_lev = calculate_returns(cf_severe, initial_portfolio_value, 0.85, 0.065)

# ============================================================================
# 8. PRINT RESULTS
# ============================================================================
print("\n" + "="*80)
print("SCENARIO ANALYSIS RESULTS")
print("="*80)

print("\nUNLEVERED RETURNS:")
print(f"\nBase Case:")
print(f"  IRR: {returns_base_unlev['irr_annual']*100:.2f}%")
print(f"  MOIC: {returns_base_unlev['moic']:.2f}x")
print(f"  WAL: {returns_base_unlev['wal_years']:.2f} years")
print(f"  Loss Rate: {returns_base_unlev['loss_rate']*100:.2f}%")

print(f"\nModerate Stress:")
print(f"  IRR: {returns_mod_unlev['irr_annual']*100:.2f}%")
print(f"  MOIC: {returns_mod_unlev['moic']:.2f}x")
print(f"  WAL: {returns_mod_unlev['wal_years']:.2f} years")
print(f"  Loss Rate: {returns_mod_unlev['loss_rate']*100:.2f}%")

print(f"\nSevere Stress:")
print(f"  IRR: {returns_sev_unlev['irr_annual']*100:.2f}%")
print(f"  MOIC: {returns_sev_unlev['moic']:.2f}x")
print(f"  WAL: {returns_sev_unlev['wal_years']:.2f} years")
print(f"  Loss Rate: {returns_sev_unlev['loss_rate']*100:.2f}%")

print("\n" + "-"*80)
print("LEVERED RETURNS (85% LTV, 6.5% Cost of Debt):")
print(f"\nBase Case:")
print(f"  Equity Investment: ${returns_base_lev['equity_amount']:,.0f}")
print(f"  IRR: {returns_base_lev['irr_annual']*100:.2f}%")
print(f"  MOIC: {returns_base_lev['moic']:.2f}x")

print(f"\nModerate Stress:")
print(f"  Equity Investment: ${returns_mod_lev['equity_amount']:,.0f}")
print(f"  IRR: {returns_mod_lev['irr_annual']*100:.2f}%")
print(f"  MOIC: {returns_mod_lev['moic']:.2f}x")

print(f"\nSevere Stress:")
print(f"  Equity Investment: ${returns_sev_lev['equity_amount']:,.0f}")
print(f"  IRR: {returns_sev_lev['irr_annual']*100:.2f}%")
print(f"  MOIC: {returns_sev_lev['moic']:.2f}x")

# Save results
results = {
    'base_unlevered': returns_base_unlev,
    'moderate_unlevered': returns_mod_unlev,
    'severe_unlevered': returns_sev_unlev,
    'base_levered': returns_base_lev,
    'moderate_levered': returns_mod_lev,
    'severe_levered': returns_sev_lev
}

import pickle
with open('analysis_results.pkl', 'wb') as f:
    pickle.dump({
        'results': results,
        'cashflows': {'base': cf_base, 'moderate': cf_moderate, 'severe': cf_severe}
    }, f)

print("\n" + "="*80)
print("ANALYSIS COMPLETE - Results saved to analysis_results.pkl")
print("="*80)
