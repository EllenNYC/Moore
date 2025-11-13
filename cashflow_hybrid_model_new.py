#!/usr/bin/env python3
"""
Cashflow Projection Using Deterministic Probability-Weighted Approach
- Uses logic from calculate_new_loan_cum_default.py (NO Monte Carlo simulation)
- Regression for Current → D1-29 and Current → Payoff
- Empirical Program x Term matrices for other transitions
- 6-month extension for delinquent loans at maturity
"""

import pandas as pd
import numpy as np
import pickle
from scipy import optimize
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("CASHFLOW PROJECTION - DETERMINISTIC PROBABILITY-WEIGHTED MODEL")
print("="*80)

# ============================================================================
# 1. LOAD MODELS AND DATA
# ============================================================================
print("\n1. Loading hybrid transition models...")

with open('hybrid_transition_models.pkl', 'rb') as f:
    models = pickle.load(f)

model_d1_29 = models['model_d1_29']
scaler_d1_29 = models['scaler_d1_29']
model_prepay = models['model_prepay']
scaler_prepay = models['scaler_prepay']
transition_matrices = models['transition_matrices']
programs = models['programs']
term_buckets = models['term_buckets']
feature_cols_d1_29 = models['feature_cols_d1_29']
feature_cols_prepay = models['feature_cols_prepay']
numeric_features_d1_29 = models['numeric_features_d1_29']
numeric_features_prepay = models['numeric_features_prepay']

print(f"  Loaded regression models (AUC D1-29: {models['auc_d1_29']:.3f}, Payoff: {models['auc_prepay']:.3f})")
print(f"  Loaded {len(transition_matrices)} empirical matrices (Program x Term)")

# Load loan data from enhanced dataset
print("  Loading loan_performance_enhanced.csv...")
loan_perf = pd.read_csv('loan_performance_enhanced.csv')
loan_perf['report_date'] = pd.to_datetime(loan_perf['report_date'])
loan_perf['disbursement_d'] = pd.to_datetime(loan_perf['disbursement_d'])

# Get the most recent snapshot of each loan for portfolio projection
loan_tape = loan_perf.sort_values('report_date').groupby('display_id').last().reset_index()

print(f"  Loaded {len(loan_tape):,} unique loans from enhanced dataset")

# ============================================================================
# 2. DETERMINISTIC CASHFLOW PROJECTION FUNCTION
# ============================================================================

def project_cashflows_deterministic(portfolio_df, model_d1_29, scaler_d1_29, model_prepay, scaler_prepay,
                                   transition_matrices, feature_cols_d1_29, feature_cols_prepay,
                                   recovery_rate=0.15, months=60, stress_d1_29_mult=1.0, stress_co_mult=1.0):
    """
    Project cashflows using deterministic probability-weighted approach.
    NO Monte Carlo simulation - tracks probability distribution across states.

    Parameters:
    -----------
    stress_d1_29_mult : float, default 1.0
        Multiplier for D1-29 transition probability (e.g., 1.2 = 20% increase)
    stress_co_mult : float, default 1.0
        Multiplier for charge-off transition probability (e.g., 1.5 = 50% increase)
    """

    n_loans = len(portfolio_df)

    # Reset index to ensure contiguous indexing
    portfolio_df = portfolio_df.reset_index(drop=True)

    # Initialize state probabilities and UPB for each loan
    # Each loan starts 100% in CURRENT state with full UPB
    loan_state_probs = []
    loan_state_upb = []
    loan_ever_d30 = []

    # CRITICAL: Track loan age for each loan (starts from current loan_age_months)
    loan_ages = portfolio_df['loan_age_months'].values.copy()

    for idx, loan in portfolio_df.iterrows():
        state_probs = {
            'CURRENT': 1.0,
            'D1_29': 0.0,
            'D30_59': 0.0,
            'D60_89': 0.0,
            'D90_119': 0.0,
            'D120_PLUS': 0.0,
            'CHARGED_OFF': 0.0,
            'PAID_OFF': 0.0
        }

        state_upb = {
            'CURRENT': loan['upb'],
            'D1_29': 0.0,
            'D30_59': 0.0,
            'D60_89': 0.0,
            'D90_119': 0.0,
            'D120_PLUS': 0.0,
            'CHARGED_OFF': 0.0,
            'PAID_OFF': 0.0
        }

        loan_state_probs.append(state_probs)
        loan_state_upb.append(state_upb)
        loan_ever_d30.append(0)

    # Calculate monthly payments
    monthly_payments = []
    for idx, loan in portfolio_df.iterrows():
        r = loan['int_rate'] / 12
        n = loan['loan_term']
        amount = loan['approved_amount']

        if r > 0 and n > 0:
            payment = amount * (r * (1 + r)**n) / ((1 + r)**n - 1)
        else:
            payment = amount / max(n, 1)

        monthly_payments.append(payment)

    monthly_payments = np.array(monthly_payments)

    # Track cashflows
    cashflows = []

    all_states = ['CURRENT', 'D1_29', 'D30_59', 'D60_89', 'D90_119', 'D120_PLUS', 'CHARGED_OFF', 'PAID_OFF']
    delinq_states = ['D1_29', 'D30_59', 'D60_89', 'D90_119', 'D120_PLUS']

    # Determine max months (longest loan term + 6 months extension)
    max_loan_term = portfolio_df['loan_term'].max()
    max_months = min(int(max_loan_term + 6), months)

    for month in range(1, max_months + 1):
        # Initialize monthly aggregates
        month_interest = 0.0
        month_scheduled_principal = 0.0
        month_payoffs = 0.0
        month_defaults = 0.0
        month_recoveries = 0.0
        month_ending_balance = 0.0
        month_active_loans = 0.0
        month_current_count = 0.0
        month_chargedoff_count = 0.0
        month_paidoff_count = 0.0

        # Process each loan
        for loan_idx, loan in portfolio_df.iterrows():
            idx = loan_idx

            loan_term = loan['loan_term']
            amount = loan['approved_amount']
            monthly_rate = loan['int_rate'] / 12
            monthly_payment = monthly_payments[idx]
            program = loan['program']
            fico = loan['fico_score']

            # Get current state distribution
            state_probs = loan_state_probs[idx]
            state_upb = loan_state_upb[idx]
            ever_d30 = loan_ever_d30[idx]

            # Calculate CURRENT UPB before payment
            current_upb_before = state_upb['CURRENT']

            # Calculate scheduled principal for CURRENT loans (only during term)
            if month <= loan_term and current_upb_before > 0:
                if monthly_rate > 0:
                    interest = current_upb_before * monthly_rate
                    sched_principal = min(monthly_payment - interest, current_upb_before)
                else:
                    interest = 0
                    sched_principal = monthly_payment

                current_upb_after = max(0, current_upb_before - sched_principal)
            else:
                # Post-maturity: no more payments
                interest = 0
                sched_principal = 0
                current_upb_after = current_upb_before

            # New state distribution
            new_state_probs = {s: 0.0 for s in all_states}
            new_state_upb = {s: 0.0 for s in all_states}

            # Process each current state
            for current_state, prob in state_probs.items():
                if prob < 0.0001:
                    continue

                current_state_upb = state_upb[current_state]

                # Terminal states stay terminal
                if current_state in ['CHARGED_OFF', 'PAID_OFF']:
                    new_state_probs[current_state] += prob
                    new_state_upb[current_state] += 0
                    continue

                # CURRENT state: use regression models
                if current_state == 'CURRENT':
                    # Predict transitions using regression models (both during and after term)
                    # The models are trained on actual data and will learn post-maturity behavior
                    p_d1_29 = predict_d1_29(month, program, loan_term, fico, amount, ever_d30,
                                           model_d1_29, scaler_d1_29, feature_cols_d1_29)
                    # Apply stress to D1-29 probability
                    p_d1_29 = min(p_d1_29 * stress_d1_29_mult, 0.99)

                    p_prepay = predict_prepay(month, program, loan_term, fico, current_upb_before,
                                             model_prepay, scaler_prepay, feature_cols_prepay)
                    p_stay_current = max(0, 1.0 - p_d1_29 - p_prepay)

                    # Update probabilities
                    new_state_probs['D1_29'] += prob * p_d1_29
                    new_state_probs['PAID_OFF'] += prob * p_prepay
                    new_state_probs['CURRENT'] += prob * p_stay_current

                    # Track payoffs from CURRENT
                    payoff_amount = prob * p_prepay * current_upb_before
                    month_payoffs += payoff_amount

                    # Update UPB
                    # For post-maturity loans, current_upb_after = current_upb_before (no payment)
                    new_state_upb['D1_29'] += prob * p_d1_29 * current_upb_before  # Freeze at pre-payment
                    new_state_upb['PAID_OFF'] += 0  # Paid off, no remaining balance
                    new_state_upb['CURRENT'] += prob * p_stay_current * current_upb_after  # After payment

                # Delinquent states: use empirical matrices
                elif current_state in delinq_states:
                    if month <= loan_term + 6:
                        # Get transition matrix
                        trans_probs = get_transition_probs(current_state, program, loan_term,
                                                          transition_matrices, all_states)

                        # Apply stress to charge-off probability
                        if stress_co_mult != 1.0:
                            orig_co = trans_probs.get('CHARGED_OFF', 0)
                            co_prob = min(orig_co * stress_co_mult, 0.99)
                            prob_increase = co_prob - orig_co

                            # Reduce other probabilities proportionally
                            total_other = sum(trans_probs.values()) - orig_co
                            if total_other > 0:
                                for state in trans_probs:
                                    if state != 'CHARGED_OFF':
                                        trans_probs[state] = max(0, trans_probs[state] - (trans_probs[state] / total_other) * prob_increase)
                                trans_probs['CHARGED_OFF'] = co_prob

                        for next_state, trans_prob in trans_probs.items():
                            new_state_probs[next_state] += prob * trans_prob

                            upb_moved = trans_prob * current_state_upb

                            if next_state == 'CHARGED_OFF':
                                # Track default
                                month_defaults += upb_moved
                                month_recoveries += upb_moved * recovery_rate
                                new_state_upb[next_state] += 0
                            elif next_state == 'PAID_OFF':
                                month_payoffs += upb_moved
                                new_state_upb[next_state] += 0
                            elif next_state == 'CURRENT':
                                new_state_upb[next_state] += prob * trans_prob * current_upb_after
                            else:
                                new_state_upb[next_state] += upb_moved
                    else:
                        # After extension: remaining delinquents charge off
                        new_state_probs['CHARGED_OFF'] += prob
                        month_defaults += current_state_upb
                        month_recoveries += current_state_upb * recovery_rate
                        new_state_upb['CHARGED_OFF'] += 0

            # Update loan state
            loan_state_probs[idx] = new_state_probs
            loan_state_upb[idx] = new_state_upb

            # Update ever_d30 flag
            if sum(new_state_probs.get(s, 0) for s in ['D30_59', 'D60_89', 'D90_119', 'D120_PLUS']) > 0.01:
                loan_ever_d30[idx] = 1

            # Accumulate monthly totals
            if state_probs['CURRENT'] > 0 and month <= loan_term:
                month_interest += state_probs['CURRENT'] * interest
                month_scheduled_principal += state_probs['CURRENT'] * sched_principal

            # Count active loans (probability-weighted)
            active_prob = sum(new_state_probs[s] for s in all_states if s not in ['CHARGED_OFF', 'PAID_OFF'])
            month_active_loans += active_prob

            month_current_count += new_state_probs['CURRENT']
            month_chargedoff_count += new_state_probs['CHARGED_OFF']
            month_paidoff_count += new_state_probs['PAID_OFF']

            # Ending balance
            month_ending_balance += sum(new_state_upb.values())

        # Record monthly cashflow
        total_inflow = month_interest + month_scheduled_principal + month_payoffs + month_recoveries
        net_loss = month_defaults - month_recoveries

        cashflows.append({
            'month': month,
            'interest': month_interest,
            'scheduled_principal': month_scheduled_principal,
            'payoffs': month_payoffs,
            'defaults': month_defaults,
            'recoveries': month_recoveries,
            'total_inflow': total_inflow,
            'net_loss': net_loss,
            'ending_balance': month_ending_balance,
            'active_loans': month_active_loans,
            'current_count': month_current_count,
            'chargedoff_count': month_chargedoff_count,
            'paidoff_count': month_paidoff_count
        })

    return pd.DataFrame(cashflows)


def predict_d1_29(loan_age, program, loan_term, fico, amount, ever_d30,
                 model, scaler, feature_cols):
    """Predict D1-29 probability using regression model"""

    # Age bucket
    if loan_age <= 1:
        age_bucket = '0-1m'
    elif loan_age <= 3:
        age_bucket = '2-3m'
    elif loan_age <= 6:
        age_bucket = '4-6m'
    elif loan_age <= 12:
        age_bucket = '7-12m'
    elif loan_age <= 18:
        age_bucket = '13-18m'
    elif loan_age <= 24:
        age_bucket = '19-24m'
    else:
        age_bucket = '24m+'

    # FICO bucket
    if fico < 620:
        fico_bucket = 'fico_<620'
    elif fico < 660:
        fico_bucket = 'fico_620-659'
    elif fico < 700:
        fico_bucket = 'fico_660-699'
    elif fico < 740:
        fico_bucket = 'fico_700-739'
    else:
        fico_bucket = 'fico_740+'

    # Amount bucket
    if amount <= 2000:
        amount_bucket = 'amt_0-2k'
    elif amount <= 4000:
        amount_bucket = 'amt_2-4k'
    elif amount <= 6000:
        amount_bucket = 'amt_4-6k'
    elif amount <= 8000:
        amount_bucket = 'amt_6-8k'
    else:
        amount_bucket = 'amt_8k+'

    # Build features
    features = {'ever_D30': ever_d30}

    for prog in ['P2', 'P3']:
        features[f'program_{prog}'] = int(program == prog)

    for bucket in ['2-3m', '4-6m', '7-12m', '13-18m', '19-24m', '24m+']:
        features[f'age_{bucket}'] = int(age_bucket == bucket)

    # CRITICAL: FICO and amount buckets need double prefix to match model training
    for bucket in ['fico_620-659', 'fico_660-699', 'fico_700-739', 'fico_740+', 'fico_<620']:
        features[f'fico_{bucket}'] = int(fico_bucket == bucket)

    for bucket in ['amt_2-4k', 'amt_4-6k', 'amt_6-8k', 'amt_8k+']:
        features[f'amt_{bucket}'] = int(amount_bucket == bucket)

    for term in [6, 12, 24, 36, 60]:
        features[f'term_{term}'] = int(loan_term == term)

    # Create feature array
    X = [features.get(col, 0) for col in feature_cols]
    X = np.array(X).reshape(1, -1)
    X_scaled = scaler.transform(X)

    return model.predict_proba(X_scaled)[0, 1]


def predict_prepay(loan_age, program, loan_term, fico, upb,
                  model, scaler, feature_cols):
    """Predict prepayment probability using regression model"""

    # Age bucket
    if loan_age <= 1:
        age_bucket = '0-1m'
    elif loan_age <= 3:
        age_bucket = '2-3m'
    elif loan_age <= 6:
        age_bucket = '4-6m'
    elif loan_age <= 12:
        age_bucket = '7-12m'
    elif loan_age <= 18:
        age_bucket = '13-18m'
    elif loan_age <= 24:
        age_bucket = '19-24m'
    else:
        age_bucket = '24m+'

    # FICO bucket
    if fico < 620:
        fico_bucket = 'fico_<620'
    elif fico < 660:
        fico_bucket = 'fico_620-659'
    elif fico < 700:
        fico_bucket = 'fico_660-699'
    elif fico < 740:
        fico_bucket = 'fico_700-739'
    else:
        fico_bucket = 'fico_740+'

    # UPB bucket
    if upb <= 1000:
        upb_bucket = 'upb_0-1k'
    elif upb <= 2500:
        upb_bucket = 'upb_1-2.5k'
    elif upb <= 5000:
        upb_bucket = 'upb_2.5-5k'
    elif upb <= 7500:
        upb_bucket = 'upb_5-7.5k'
    else:
        upb_bucket = 'upb_7.5k+'

    # Build features
    features = {}

    for prog in ['P2', 'P3']:
        features[f'program_{prog}'] = int(program == prog)

    for bucket in ['2-3m', '4-6m', '7-12m', '13-18m', '19-24m', '24m+']:
        features[f'age_{bucket}'] = int(age_bucket == bucket)

    # CRITICAL: FICO buckets need double prefix to match model training
    for bucket in ['fico_620-659', 'fico_660-699', 'fico_700-739', 'fico_740+', 'fico_<620']:
        features[f'fico_{bucket}'] = int(fico_bucket == bucket)

    for term in [6, 12, 24, 36, 60]:
        features[f'term_{term}'] = int(loan_term == term)

    # UPB buckets also need double prefix
    for bucket in ['upb_0-1k', 'upb_1-2.5k', 'upb_2.5-5k', 'upb_5-7.5k', 'upb_7.5k+']:
        features[f'upb_{bucket}'] = int(upb_bucket == bucket)

    # Create feature array
    X = [features.get(col, 0) for col in feature_cols]
    X = np.array(X).reshape(1, -1)
    X_scaled = scaler.transform(X)

    return model.predict_proba(X_scaled)[0, 1]


def get_transition_probs(from_state, program, loan_term, transition_matrices, all_states):
    """Get transition probabilities from empirical matrices"""

    if from_state not in transition_matrices:
        probs = {state: 0.0 for state in all_states}
        probs[from_state] = 1.0
        return probs

    matrix = transition_matrices[from_state]
    key = (program, loan_term)

    if key in matrix:
        return matrix[key]
    else:
        # Fall back to program average
        program_keys = [k for k in matrix.keys() if k[0] == program]
        if program_keys:
            avg_probs = {state: 0.0 for state in all_states}
            for k in program_keys:
                for state, prob in matrix[k].items():
                    avg_probs[state] += prob / len(program_keys)
            return avg_probs
        else:
            # Fall back to overall average
            avg_probs = {state: 0.0 for state in all_states}
            for k in matrix.keys():
                for state, prob in matrix[k].items():
                    avg_probs[state] += prob / len(matrix)
            return avg_probs


# ============================================================================
# 3. RETURN CALCULATION
# ============================================================================

def calculate_irr_newton(cashflows, initial_investment):
    all_cf = [-initial_investment] + list(cashflows)
    def npv(rate):
        return sum([cf / (1 + rate) ** i for i, cf in enumerate(all_cf)])
    try:
        irr_monthly = optimize.newton(npv, 0.001, maxiter=100, tol=1e-6)
        return (1 + irr_monthly) ** 12 - 1
    except:
        total_return = sum(cashflows)
        return (total_return / initial_investment) - 1 if initial_investment > 0 else 0

def calculate_returns(cashflows, initial_investment, leverage_ratio=0.0, cost_of_debt=0.0):
    portfolio_value = initial_investment
    debt_amount = portfolio_value * leverage_ratio
    equity_amount = portfolio_value * (1 - leverage_ratio)

    monthly_debt_rate = cost_of_debt / 12

    equity_cashflows = []
    outstanding_debt = debt_amount

    for _, row in cashflows.iterrows():
        interest_expense = outstanding_debt * monthly_debt_rate
        principal_collected = row['scheduled_principal'] + row['payoffs'] + row['recoveries']

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
        'irr': irr,
        'moic': moic,
        'wal_years': wal,
        'loss_rate': loss_rate,
        'total_interest': cashflows['interest'].sum(),
        'total_losses': cashflows['net_loss'].sum()
    }

# ============================================================================
# 4. RUN SCENARIOS
# ============================================================================
print("\n2. Running scenario analysis...")

sample_size = min(10000, len(loan_tape))

# Filter for September 2023 originations
sep_2023_start = pd.to_datetime('2023-09-01')
sep_2023_end = pd.to_datetime('2023-09-30')
oct_2023 = pd.to_datetime('2023-10-01')

active_loans = loan_tape[
    (loan_tape['disbursement_d'] >= sep_2023_start) &
    (loan_tape['disbursement_d'] <= sep_2023_end) &
    (loan_tape['upb'] > 100) &
    (loan_tape['report_date'] >= oct_2023)
]

required_cols = ['approved_amount', 'int_rate', 'loan_term', 'fico_score', 'program',
                 'upb', 'paid_principal', 'paid_interest', 'ever_D30', 'ever_D60', 'ever_D90']

# portfolio_sample = active_loans.dropna(subset=required_cols).sample(
#     n=min(sample_size, len(active_loans)), random_state=42
# )
portfolio_sample = active_loans.dropna(subset=required_cols)
# Use current UPB as initial value
price_percent = 1 - portfolio_sample['mdr'] + 0.005
initial_value = (portfolio_sample['upb'] * price_percent).sum()

print(f"\n  Portfolio: {len(portfolio_sample):,} active loans (2023Q3 originations)")
print(f"  Origination Date Range: {portfolio_sample['disbursement_d'].min().strftime('%Y-%m-%d')} to {portfolio_sample['disbursement_d'].max().strftime('%Y-%m-%d')}")
print(f"  Current UPB: ${initial_value:,.0f}")
print(f"  Original Amount: ${portfolio_sample['approved_amount'].sum():,.0f}")

# Define scenarios with stress testing
scenarios = {
    'Base Case': {'recovery': 0, 'stress_d1_29': 1.0, 'stress_co': 1.0},
    'Moderate Stress': {'recovery': 0, 'stress_d1_29': 1.2, 'stress_co': 1.5},
    'Severe Stress': {'recovery': 0, 'stress_d1_29': 1.6, 'stress_co': 2.5}
}

results = {}
cashflow_results = {}

for scenario_name, params in scenarios.items():
    print(f"\n  {scenario_name}:")
    print(f"    Recovery: {params['recovery']*100:.0f}%")
    print(f"    D1-29 Stress: {params['stress_d1_29']:.1f}x, Charge-Off Stress: {params['stress_co']:.1f}x")

    cf = project_cashflows_deterministic(
        portfolio_sample,
        model_d1_29, scaler_d1_29,
        model_prepay, scaler_prepay,
        transition_matrices,
        feature_cols_d1_29, feature_cols_prepay,
        recovery_rate=params['recovery'],
        months=60,
        stress_d1_29_mult=params['stress_d1_29'],
        stress_co_mult=params['stress_co']
    )

    cashflow_results[scenario_name] = cf

    unlev = calculate_returns(cf, initial_value, 0.0, 0.0)
    # assume cost_of_debt = 3.6%+1.5% = 5.1%
    lev = calculate_returns(cf, initial_value, 0.85, 0.051)

    results[scenario_name] = {'unlevered': unlev, 'levered': lev}

    print(f"    Unlevered IRR: {unlev['irr']*100:.2f}%, Loss: {unlev['loss_rate']*100:.1f}%")
    print(f"    Levered IRR: {lev['irr']*100:.2f}%")

# ============================================================================
# 5. SUMMARY
# ============================================================================
print("\n" + "="*80)
print("RESULTS SUMMARY - DETERMINISTIC HYBRID MODEL")
print("="*80)

summary_data = []
for scenario_name in scenarios.keys():
    unlev = results[scenario_name]['unlevered']
    lev = results[scenario_name]['levered']

    summary_data.append({
        'Scenario': scenario_name,
        'investment ($)': f"${unlev['portfolio_value']/1e6:.2f}M",
        'Unlevered IRR': f"{unlev['irr']*100:.1f}%",
        'Unlevered MOIC': f"{unlev['moic']:.2f}x",
        'Levered IRR': f"{lev['irr']*100:.1f}%",
        'Levered MOIC': f"{lev['moic']:.2f}x",
        'Loss Rate': f"{unlev['loss_rate']*100:.1f}%",
        'WAL': f"{unlev['wal_years']:.1f}y"
    })

summary_df = pd.DataFrame(summary_data)
print("\n" + summary_df.to_string(index=False))

# Save
with open('cashflow_deterministic_results.pkl', 'wb') as f:
    pickle.dump({'results': results, 'cashflows': cashflow_results, 'summary': summary_df}, f)

print("\n  Results saved to cashflow_deterministic_results.pkl")

# ============================================================================
# 6. CASHFLOW VISUALIZATION - STACKED COLUMN CHARTS
# ============================================================================
print("\n3. Creating cashflow visualizations...")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Create a combined comparison chart (all scenarios side-by-side)
n_scenarios = len(scenarios)
fig, axes = plt.subplots(1, n_scenarios, figsize=(20, 7))
if n_scenarios == 1:
    axes = [axes]

for idx, (scenario_name, cf) in enumerate(cashflow_results.items()):
    ax = axes[idx]

    months = cf['month'].values
    interest_payments = cf['interest'].values
    principal_payments = cf['scheduled_principal'].values
    payoffs = cf['payoffs'].values
    defaults = cf['defaults'].values

    # Calculate totals
    total_principal = principal_payments.sum()
    total_interest = interest_payments.sum()
    total_payoff = payoffs.sum()
    total_default = defaults.sum()
    total_gross_cf = total_principal + total_interest + total_payoff
    total_net_cf = total_gross_cf - total_default
    moic = total_net_cf / initial_value

    # Stacked bars
    ax.bar(months, interest_payments, label='Interest', color='forestgreen', alpha=0.8)
    ax.bar(months, principal_payments, bottom=interest_payments,
           label='Principal', color='steelblue', alpha=0.8)
    ax.bar(months, payoffs, bottom=interest_payments + principal_payments,
           label='Payoff', color='orange', alpha=0.8)
    ax.bar(months, -defaults, label='Default', color='crimson', alpha=0.8)

    # Add text box with totals (including initial investment, net CF, and MOIC)
    textstr = f'Initial Inv: ${initial_value/1e6:.2f}M\n'
    textstr += f'─────────────────────\n'
    textstr += f'Principal: ${total_principal/1e6:.2f}M\n'
    textstr += f'Interest: ${total_interest/1e6:.2f}M\n'
    textstr += f'Payoff: ${total_payoff/1e6:.2f}M\n'
    textstr += f'Default: ${total_default/1e6:.2f}M\n'
    textstr += f'─────────────────────\n'
    textstr += f'Net CF: ${total_net_cf/1e6:.2f}M\n'
    textstr += f'MOIC: {moic:.2f}x'

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.85)
    ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right', bbox=props, family='monospace')

    ax.set_xlabel('Month', fontsize=10, fontweight='bold')
    ax.set_ylabel('Cashflow ($)', fontsize=10, fontweight='bold')
    ax.set_title(scenario_name, fontsize=12, fontweight='bold')
    if idx == 0:
        # Create custom legend with colored markers
        legend_elements = [
            Rectangle((0, 0), 1, 1, fc='forestgreen', alpha=0.8, label='Interest'),
            Rectangle((0, 0), 1, 1, fc='steelblue', alpha=0.8, label='Principal'),
            Rectangle((0, 0), 1, 1, fc='orange', alpha=0.8, label='Payoff'),
            Rectangle((0, 0), 1, 1, fc='crimson', alpha=0.8, label='Default')
        ]
        ax.legend(handles=legend_elements, fontsize=9, loc='upper left', framealpha=0.9)
    ax.grid(alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linewidth=0.8, linestyle='-')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))

fig.suptitle('Cashflow Breakdown Comparison - Deterministic Model', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('cashflow_breakdown_comparison_deterministic.png', dpi=150, bbox_inches='tight')
print("  ✓ Saved cashflow_breakdown_comparison_deterministic.png")
plt.close()

print("\n" + "="*80)
print("DETERMINISTIC CASHFLOW ANALYSIS COMPLETE")
print("="*80)
print("\nMETHODOLOGY:")
print("  • NO Monte Carlo simulation")
print("  • Tracks probability distribution across states for each loan")
print("  • Regression models for CURRENT transitions")
print("  • Empirical Program × Term matrices for delinquent transitions")
print("  • 6-month post-maturity extension for delinquent loans")
print("  • Expected value calculation (not sampled outcomes)")

print("\n" + "="*80)
