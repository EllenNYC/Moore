#!/usr/bin/env python3
"""
Consumer Credit Portfolio Analysis - WITH PROGRAM FEATURE
Moore Capital Case Study
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from scipy import optimize
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("CONSUMER CREDIT ANALYSIS - WITH PROGRAM FEATURE IN MODELS")
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

# Clean performance data
loan_performance['report_date'] = pd.to_datetime(loan_performance['report_date'], errors='coerce')

print(f"  Loaded {len(loan_tape):,} loans")
print(f"  Total portfolio value: ${loan_tape['approved_amount'].sum():,.0f}")
print(f"  Program distribution:")
for prog, count in loan_tape['program'].value_counts().items():
    print(f"    {prog}: {count:,} ({count/len(loan_tape)*100:.1f}%)")

# Get final status
loan_final_status = loan_performance.sort_values('report_date').groupby('display_id').last().reset_index()
loan_analysis = loan_tape.merge(loan_final_status[['display_id', 'loan_status']], on='display_id', how='left')

# Create outcome flags
loan_analysis['defaulted'] = loan_analysis['loan_status'].isin(['CHARGED_OFF', 'WRITTEN_OFF']).astype(int)
loan_analysis['prepaid'] = loan_analysis['loan_status'].isin(['PAID_OFF', 'SATISFIED']).astype(int)

print(f"\n  Default rate: {loan_analysis['defaulted'].mean() * 100:.2f}%")
print(f"  Prepayment rate: {loan_analysis['prepaid'].mean() * 100:.2f}%")

# ============================================================================
# 2. BUILD DEFAULT MODEL WITH PROGRAM
# ============================================================================
print("\n2. Building default probability model (with program)...")

# Prepare features
numeric_features = ['fico_score', 'approved_amount', 'loan_term', 'int_rate', 'mdr']
modeling_data = loan_analysis[loan_analysis['loan_status'] != 'CURRENT'].copy()

# One-hot encode program
program_dummies = pd.get_dummies(modeling_data['program'], prefix='program', drop_first=True)

# Combine features
X_default = pd.concat([modeling_data[numeric_features], program_dummies], axis=1)
y_default = modeling_data['defaulted']

# Remove missing values
valid_idx = X_default.notna().all(axis=1) & y_default.notna()
X_default_clean = X_default[valid_idx]
y_default_clean = y_default[valid_idx]

# Split and train
X_train, X_test, y_train, y_test = train_test_split(
    X_default_clean, y_default_clean, test_size=0.3, random_state=42, stratify=y_default_clean
)

scaler_default = StandardScaler()
X_train_scaled = scaler_default.fit_transform(X_train)
X_test_scaled = scaler_default.transform(X_test)

default_model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
default_model.fit(X_train_scaled, y_train)

test_auc = roc_auc_score(y_test, default_model.predict_proba(X_test_scaled)[:, 1])
print(f"  Default model AUC: {test_auc:.4f}")

# Feature importance
feature_cols_default = list(X_default_clean.columns)
feature_importance_default = pd.DataFrame({
    'Feature': feature_cols_default,
    'Coefficient': default_model.coef_[0]
}).sort_values('Coefficient', key=abs, ascending=False)
print(f"\n  Top 5 features:")
for idx, row in feature_importance_default.head(5).iterrows():
    print(f"    {row['Feature']:20s}: {row['Coefficient']:+.3f}")

# ============================================================================
# 3. BUILD PREPAYMENT MODEL WITH PROGRAM
# ============================================================================
print("\n3. Building prepayment probability model (with program)...")

prepay_modeling_data = modeling_data[modeling_data['defaulted'] == 0].copy()

# One-hot encode program
program_dummies_prepay = pd.get_dummies(prepay_modeling_data['program'], prefix='program', drop_first=True)

# Combine features
X_prepay = pd.concat([prepay_modeling_data[numeric_features], program_dummies_prepay], axis=1)
y_prepay = prepay_modeling_data['prepaid']

# Remove missing values
valid_idx_p = X_prepay.notna().all(axis=1) & y_prepay.notna()
X_prepay_clean = X_prepay[valid_idx_p]
y_prepay_clean = y_prepay[valid_idx_p]

# Split and train
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
    X_prepay_clean, y_prepay_clean, test_size=0.3, random_state=42, stratify=y_prepay_clean
)

scaler_prepay = StandardScaler()
X_train_p_scaled = scaler_prepay.fit_transform(X_train_p)
X_test_p_scaled = scaler_prepay.transform(X_test_p)

prepay_model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
prepay_model.fit(X_train_p_scaled, y_train_p)

test_auc_p = roc_auc_score(y_test_p, prepay_model.predict_proba(X_test_p_scaled)[:, 1])
print(f"  Prepayment model AUC: {test_auc_p:.4f}")

# Feature importance
feature_cols_prepay = list(X_prepay_clean.columns)
feature_importance_prepay = pd.DataFrame({
    'Feature': feature_cols_prepay,
    'Coefficient': prepay_model.coef_[0]
}).sort_values('Coefficient', key=abs, ascending=False)
print(f"\n  Top 5 features:")
for idx, row in feature_importance_prepay.head(5).iterrows():
    print(f"    {row['Feature']:20s}: {row['Coefficient']:+.3f}")

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
# 5. RETURN CALCULATION FUNCTION
# ============================================================================

def calculate_irr_newton(cashflows, initial_investment):
    """Calculate IRR using Newton's method."""
    all_cf = [-initial_investment] + list(cashflows)

    def npv(rate):
        return sum([cf / (1 + rate) ** i for i, cf in enumerate(all_cf)])

    try:
        irr_monthly = optimize.newton(npv, 0.001, maxiter=100, tol=1e-6)
        return (1 + irr_monthly) ** 12 - 1
    except:
        total_return = sum(cashflows)
        return (total_return / initial_investment) - 1

def calculate_returns(cashflows, initial_investment, leverage_ratio=0.0, cost_of_debt=0.0):
    """Calculate returns with leverage."""

    portfolio_value = initial_investment
    debt_amount = portfolio_value * leverage_ratio
    equity_amount = portfolio_value * (1 - leverage_ratio)

    monthly_debt_rate = cost_of_debt / 12

    equity_cashflows = []
    outstanding_debt = debt_amount

    for _, row in cashflows.iterrows():
        interest_expense = outstanding_debt * monthly_debt_rate
        principal_collected = row['scheduled_principal'] + row['prepayments'] + row['recoveries']

        debt_paydown = min(principal_collected, outstanding_debt)
        outstanding_debt -= debt_paydown

        eq_cf = row['interest'] - interest_expense - row['net_loss'] + (principal_collected - debt_paydown)
        equity_cashflows.append(eq_cf)

    irr = calculate_irr_newton(equity_cashflows, equity_amount)
    total_returned = sum(equity_cashflows)
    moic = total_returned / equity_amount if equity_amount > 0 else 0

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
# 6. PREPARE PORTFOLIO FOR PROJECTIONS
# ============================================================================
print("\n4. Preparing portfolio for cashflow projections...")

# Get full portfolio with all features needed for prediction
portfolio_data = loan_tape.dropna(subset=['approved_amount', 'int_rate', 'loan_term', 'fico_score', 'mdr', 'program'])

# Create same feature structure as training
program_dummies_portfolio = pd.get_dummies(portfolio_data['program'], prefix='program', drop_first=True)
X_portfolio_default = pd.concat([portfolio_data[numeric_features], program_dummies_portfolio], axis=1)

# Ensure all columns from training are present
for col in feature_cols_default:
    if col not in X_portfolio_default.columns:
        X_portfolio_default[col] = 0
X_portfolio_default = X_portfolio_default[feature_cols_default]  # Reorder to match training

# Predict default probabilities
X_portfolio_default_scaled = scaler_default.transform(X_portfolio_default)
default_probabilities = default_model.predict_proba(X_portfolio_default_scaled)[:, 1]

# For prepayment, use same approach
X_portfolio_prepay = pd.concat([portfolio_data[numeric_features], program_dummies_portfolio], axis=1)
for col in feature_cols_prepay:
    if col not in X_portfolio_prepay.columns:
        X_portfolio_prepay[col] = 0
X_portfolio_prepay = X_portfolio_prepay[feature_cols_prepay]

X_portfolio_prepay_scaled = scaler_prepay.transform(X_portfolio_prepay)
prepay_probabilities = prepay_model.predict_proba(X_portfolio_prepay_scaled)[:, 1]

print(f"  Portfolio size: {len(portfolio_data):,} loans")
print(f"  Average predicted default probability: {default_probabilities.mean() * 100:.2f}%")
print(f"  Average predicted prepayment probability: {prepay_probabilities.mean() * 100:.2f}%")

# ============================================================================
# 7. RUN SCENARIOS
# ============================================================================
print("\n5. Running scenario analysis...")

# Sample for computation
sample_size = min(10000, len(portfolio_data))
np.random.seed(42)
sample_idx = np.random.choice(len(portfolio_data), sample_size, replace=False)
portfolio_sample = portfolio_data.iloc[sample_idx].reset_index(drop=True)
default_proba_sample = default_probabilities[sample_idx]
prepay_proba_sample = prepay_probabilities[sample_idx]

initial_value = portfolio_sample['approved_amount'].sum()

print(f"\n  Sample portfolio: {sample_size:,} loans, ${initial_value:,.0f}")

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
# 8. CALCULATE RETURNS
# ============================================================================
print("\n6. Calculating returns...")

results = {}
cashflow_results = {'base': cf_base, 'moderate': cf_moderate, 'severe': cf_severe}

scenarios_list = ['Base Case', 'Moderate Stress', 'Severe Stress']
cashflows_list = [cf_base, cf_moderate, cf_severe]

for scenario_name, cf in zip(scenarios_list, cashflows_list):
    unlev = calculate_returns(cf, initial_value, 0.0, 0.0)
    lev = calculate_returns(cf, initial_value, 0.85, 0.065)
    results[scenario_name] = {'unlevered': unlev, 'levered': lev}

# ============================================================================
# 9. PRINT RESULTS
# ============================================================================
print("\n" + "="*80)
print("SCENARIO ANALYSIS RESULTS (WITH PROGRAM FEATURE)")
print("="*80)

for scenario_name in scenarios_list:
    unlev = results[scenario_name]['unlevered']
    lev = results[scenario_name]['levered']

    print(f"\n{scenario_name}:")
    print(f"  Unlevered:")
    print(f"    IRR: {unlev['irr']*100:.2f}%")
    print(f"    MOIC: {unlev['moic']:.2f}x")
    print(f"    WAL: {unlev['wal_years']:.2f} years")
    print(f"    Loss rate: {unlev['loss_rate']*100:.2f}%")
    print(f"  Levered (85% LTV, 6.5% debt):")
    print(f"    Equity: ${lev['equity']:,.0f}")
    print(f"    IRR: {lev['irr']*100:.2f}%")
    print(f"    MOIC: {lev['moic']:.2f}x")

# Summary table
print("\n" + "="*80)
print("SUMMARY TABLE")
print("="*80)

summary_data = []
for scenario_name in scenarios_list:
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
with open('analysis_results_with_program.pkl', 'wb') as f:
    pickle.dump({
        'results': results,
        'cashflows': cashflow_results,
        'summary': summary_df,
        'models': {
            'default_model': default_model,
            'prepay_model': prepay_model,
            'scaler_default': scaler_default,
            'scaler_prepay': scaler_prepay,
            'feature_cols_default': feature_cols_default,
            'feature_cols_prepay': feature_cols_prepay,
            'default_auc': test_auc,
            'prepay_auc': test_auc_p
        }
    }, f)

print("\n" + "="*80)
print("ANALYSIS COMPLETE - Results saved to analysis_results_with_program.pkl")
print("="*80)
