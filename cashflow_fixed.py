#!/usr/bin/env python3
"""
FIXED Cashflow Projection with Hybrid Transition Model
"""

import pandas as pd
import numpy as np
import pickle
from scipy import optimize

print("="*80)
print("FIXED CASHFLOW PROJECTION - HYBRID MODEL")
print("="*80)

# Load models
with open('hybrid_transition_models.pkl', 'rb') as f:
    models = pickle.load(f)

transition_matrices = models['transition_matrices']
fico_bins = models['fico_bins']
fico_labels = models['fico_labels']
age_bins = models['age_bins']
age_labels = models['age_labels']

# Load data
loan_tape = pd.read_csv('loan tape - moore v1.0.csv')
loan_tape.columns = loan_tape.columns.str.strip()
loan_tape['mdr'] = pd.to_numeric(loan_tape['mdr'].str.rstrip('%'), errors='coerce') / 100
loan_tape['int_rate'] = pd.to_numeric(loan_tape['int_rate'].str.rstrip('%'), errors='coerce') / 100
loan_tape['approved_amount'] = pd.to_numeric(loan_tape['approved_amount'].str.replace('$', '').str.replace(',', ''), errors='coerce')

# ============================================================================
# SIMPLIFIED CASHFLOW WITH OBSERVED ROLL RATES
# ============================================================================

def project_cashflows_simple(portfolio_df, months=60, stress_factor=1.0, recovery_rate=0.15):
    """
    Simplified cashflow using observed overall roll rates
    """

    # Use empirically observed monthly rates from the data
    MONTHLY_RATES = {
        'CURRENT': {'to_d30': 0.0478 * stress_factor, 'to_prepay': 0.064, 'to_chargeoff': 0.002 * stress_factor},
        'D1_29': {'to_worse': 0.32, 'to_cure': 0.262, 'to_prepay': 0.051, 'to_chargeoff': 0.008 * stress_factor},
        'D30_59': {'to_worse': 0.673, 'to_cure': 0.070, 'to_prepay': 0.032, 'to_chargeoff': 0.009 * stress_factor},
        'D60_89': {'to_worse': 0.796, 'to_cure': 0.025, 'to_prepay': 0.018, 'to_chargeoff': 0.019 * stress_factor},
        'D90_119': {'to_chargeoff': 0.7986 * stress_factor, 'to_cure': 0.008, 'to_prepay': 0.013},
        'D120_PLUS': {'to_chargeoff': 0.8636 * stress_factor, 'to_prepay': 0.017}
    }

    n_loans = len(portfolio_df)
    cashflows = []

    balances = portfolio_df['approved_amount'].values.copy()
    monthly_rates = portfolio_df['int_rate'].values / 12
    terms = portfolio_df['loan_term'].values

    # Calculate monthly payments
    monthly_payments = np.zeros(n_loans)
    for i in range(n_loans):
        if monthly_rates[i] > 0 and terms[i] > 0:
            r = monthly_rates[i]
            n = terms[i]
            monthly_payments[i] = balances[i] * (r * (1 + r)**n) / ((1 + r)**n - 1)
        else:
            monthly_payments[i] = balances[i] / max(terms[i], 1)

    # Track states
    states = np.array(['CURRENT'] * n_loans)

    for month in range(months):
        active = ~np.isin(states, ['CHARGED_OFF', 'PAID_OFF']) & (balances > 0.01)

        if active.sum() == 0:
            break

        # Interest and principal
        interest = np.where(active, balances * monthly_rates, 0)
        sched_prin = np.where(active, np.minimum(monthly_payments - interest, balances), 0)

        # Apply transitions
        new_states = states.copy()
        charged_off_mask = np.zeros(n_loans, dtype=bool)
        paid_off_mask = np.zeros(n_loans, dtype=bool)

        for state in ['CURRENT', 'D1_29', 'D30_59', 'D60_89', 'D90_119', 'D120_PLUS']:
            state_mask = (states == state) & active

            if state_mask.sum() == 0:
                continue

            rates = MONTHLY_RATES[state]
            indices = np.where(state_mask)[0]

            # Charge-offs
            if 'to_chargeoff' in rates:
                co_prob = np.clip(rates['to_chargeoff'], 0, 0.99)
                co_this_state = np.random.random(len(indices)) < co_prob
                charged_off_mask[indices[co_this_state]] = True
                new_states[indices[co_this_state]] = 'CHARGED_OFF'

            # Prepayments (from non-charged-off)
            if 'to_prepay' in rates:
                remaining = indices[~charged_off_mask[indices]]
                if len(remaining) > 0:
                    prepay_prob = rates['to_prepay']
                    prepay_this_state = np.random.random(len(remaining)) < prepay_prob
                    paid_off_mask[remaining[prepay_this_state]] = True
                    new_states[remaining[prepay_this_state]] = 'PAID_OFF'

            # Worsen delinquency
            if 'to_worse' in rates and state != 'D120_PLUS':
                remaining = indices[~charged_off_mask[indices] & ~paid_off_mask[indices]]
                if len(remaining) > 0:
                    worse_prob = rates['to_worse']
                    worse_this_state = np.random.random(len(remaining)) < worse_prob

                    # Move to next bucket
                    next_state_map = {
                        'D1_29': 'D30_59',
                        'D30_59': 'D60_89',
                        'D60_89': 'D90_119',
                        'D90_119': 'D120_PLUS'
                    }
                    if state == 'CURRENT':
                        new_states[remaining[worse_this_state]] = 'D1_29'
                    elif state in next_state_map:
                        new_states[remaining[worse_this_state]] = next_state_map[state]

            # Cure to current
            if 'to_cure' in rates:
                remaining = indices[~charged_off_mask[indices] & ~paid_off_mask[indices]]
                if len(remaining) > 0:
                    cure_prob = rates['to_cure']
                    cure_this_state = np.random.random(len(remaining)) < cure_prob
                    new_states[remaining[cure_this_state]] = 'CURRENT'

        # Calculate cashflows from transitions
        chargeoff_amt = np.where(charged_off_mask, balances, 0)
        recovery_amt = chargeoff_amt * recovery_rate

        payoff_amt = np.where(paid_off_mask, balances, 0)

        # Update balances
        balances = np.where(charged_off_mask | paid_off_mask, 0,
                           np.maximum(balances - sched_prin, 0))

        states = new_states

        # Record
        cashflows.append({
            'month': month + 1,
            'interest': interest.sum(),
            'scheduled_principal': sched_prin.sum(),
            'prepayments': payoff_amt.sum(),
            'defaults': chargeoff_amt.sum(),
            'recoveries': recovery_amt.sum(),
            'total_inflow': interest.sum() + sched_prin.sum() + payoff_amt.sum() + recovery_amt.sum(),
            'net_loss': chargeoff_amt.sum() - recovery_amt.sum(),
            'ending_balance': balances.sum(),
            'active_loans': active.sum(),
            'current_count': (states == 'CURRENT').sum(),
            'd30_count': (states == 'D30_59').sum(),
            'd60_count': (states == 'D60_89').sum(),
            'd90_count': (states == 'D90_119').sum(),
            'charged_off_count': (states == 'CHARGED_OFF').sum(),
            'paid_off_count': (states == 'PAID_OFF').sum()
        })

    return pd.DataFrame(cashflows)

# ============================================================================
# RETURNS
# ============================================================================

def calculate_irr(cashflows, initial):
    all_cf = [-initial] + list(cashflows)
    def npv(r):
        return sum([cf / (1 + r) ** i for i, cf in enumerate(all_cf)])
    try:
        return (1 + optimize.newton(npv, 0.001, maxiter=100)) ** 12 - 1
    except:
        return (sum(cashflows) / initial) - 1 if initial > 0 else 0

def calculate_returns(cf, initial, leverage=0.0, debt_cost=0.0):
    debt = initial * leverage
    equity = initial * (1 - leverage)

    eq_cf = []
    outstanding = debt

    for _, row in cf.iterrows():
        int_exp = outstanding * (debt_cost / 12)
        prin_coll = row['scheduled_principal'] + row['prepayments'] + row['recoveries']

        paydown = min(prin_coll, outstanding)
        outstanding -= paydown

        eq = row['interest'] - int_exp - row['net_loss'] + (prin_coll - paydown)
        eq_cf.append(eq)

    irr = calculate_irr(eq_cf, equity)
    moic = sum(eq_cf) / equity if equity > 0 else 0
    wal = sum([eq_cf[i] * (i+1) for i in range(len(eq_cf))]) / sum(eq_cf) / 12 if sum(eq_cf) > 0 else 0

    return {
        'irr': irr,
        'moic': moic,
        'wal_years': wal,
        'loss_rate': cf['net_loss'].sum() / initial,
        'total_losses': cf['net_loss'].sum()
    }

# ============================================================================
# RUN SCENARIOS
# ============================================================================
print("\nRunning scenarios...")

sample_size = 10000
np.random.seed(42)
portfolio = loan_tape.dropna(subset=['approved_amount', 'int_rate', 'loan_term']).sample(sample_size, random_state=42)
initial_value = portfolio['approved_amount'].sum()

print(f"Portfolio: {sample_size:,} loans, ${initial_value:,.0f}\n")

scenarios = {
    'Base Case': {'stress': 1.0, 'recovery': 0.15},
    'Moderate Stress': {'stress': 1.5, 'recovery': 0.12},
    'Severe Stress': {'stress': 2.5, 'recovery': 0.08}
}

results = {}
cashflow_results = {}

for name, params in scenarios.items():
    print(f"{name}:")
    print(f"  Stress: {params['stress']:.1f}x, Recovery: {params['recovery']*100:.0f}%")

    np.random.seed(42)
    cf = project_cashflows_simple(portfolio, stress_factor=params['stress'], recovery_rate=params['recovery'])

    cashflow_results[name] = cf

    unlev = calculate_returns(cf, initial_value, 0.0, 0.0)
    lev = calculate_returns(cf, initial_value, 0.85, 0.065)

    results[name] = {'unlevered': unlev, 'levered': lev}

    print(f"  Unlevered IRR: {unlev['irr']*100:.2f}%, Loss: {unlev['loss_rate']*100:.1f}%")
    print(f"  Levered IRR: {lev['irr']*100:.2f}%\n")

# Summary
print("="*80)
print("RESULTS SUMMARY")
print("="*80)

summary = []
for name in scenarios.keys():
    u = results[name]['unlevered']
    l = results[name]['levered']
    summary.append({
        'Scenario': name,
        'Unlev IRR': f"{u['irr']*100:.1f}%",
        'Unlev MOIC': f"{u['moic']:.2f}x",
        'Lev IRR': f"{l['irr']*100:.1f}%",
        'Lev MOIC': f"{l['moic']:.2f}x",
        'Loss%': f"{u['loss_rate']*100:.1f}%",
        'WAL': f"{u['wal_years']:.1f}y"
    })

summary_df = pd.DataFrame(summary)
print("\n" + summary_df.to_string(index=False))

# Save
with open('fixed_cashflow_results.pkl', 'wb') as f:
    pickle.dump({'results': results, 'cashflows': cashflow_results, 'summary': summary_df}, f)

print("\n\nResults saved to fixed_cashflow_results.pkl")
print("="*80)
