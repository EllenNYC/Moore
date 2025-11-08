#!/usr/bin/env python3
"""
Cashflow Projection Model Using Delinquency Transitions
Moore Capital Case Study
"""

import pandas as pd
import numpy as np
import pickle
from scipy import optimize
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("CASHFLOW MODEL WITH DELINQUENCY TRANSITIONS")
print("="*80)

# ============================================================================
# 1. LOAD MODELS AND DATA
# ============================================================================
print("\n1. Loading transition models and data...")

# Load transition models
with open('transition_models.pkl', 'rb') as f:
    models_data = pickle.load(f)

transition_matrix = models_data['transition_matrix']
states = models_data['states']

print(f"  Loaded transition models")
print(f"  States: {states}")

# Load loan data
loan_tape = pd.read_csv('loan tape - moore v1.0.csv')
loan_tape.columns = loan_tape.columns.str.strip()
loan_tape['mdr'] = pd.to_numeric(loan_tape['mdr'].str.rstrip('%'), errors='coerce') / 100
loan_tape['int_rate'] = pd.to_numeric(loan_tape['int_rate'].str.rstrip('%'), errors='coerce') / 100
loan_tape['approved_amount'] = pd.to_numeric(loan_tape['approved_amount'].str.replace('$', '').str.replace(',', ''), errors='coerce')

# ============================================================================
# 2. TRANSITION-BASED CASHFLOW FUNCTION
# ============================================================================

def project_cashflows_with_transitions(portfolio_df, transition_matrix,
                                       recovery_rate=0.15, months=60,
                                       stress_multiplier=1.0):
    """
    Project cashflows using transition model.

    Parameters:
    - portfolio_df: DataFrame with loan characteristics
    - transition_matrix: State transition probabilities
    - recovery_rate: Recovery rate on charged-off loans
    - months: Number of months to project
    - stress_multiplier: Multiplier for charge-off transitions (1.0 = base)
    """

    n_loans = len(portfolio_df)
    cashflows = []

    # Initialize loan states
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

    # All loans start in CURRENT state
    state_indices = {state: idx for idx, state in enumerate(states)}
    loan_states = np.array([state_indices['CURRENT']] * n_loans)

    # Adjust transition matrix for stress
    stressed_matrix = transition_matrix.copy()
    if stress_multiplier != 1.0:
        # Increase charge-off transitions
        for from_state in stressed_matrix.index:
            if from_state not in ['CHARGED_OFF', 'PAID_OFF']:
                orig_co_prob = stressed_matrix.loc[from_state, 'CHARGED_OFF']
                new_co_prob = min(orig_co_prob * stress_multiplier, 0.99)

                # Redistribute remaining probability proportionally
                other_cols = [c for c in stressed_matrix.columns if c != 'CHARGED_OFF']
                remaining_prob = 1.0 - new_co_prob
                current_other_prob = stressed_matrix.loc[from_state, other_cols].sum()

                if current_other_prob > 0:
                    for col in other_cols:
                        stressed_matrix.loc[from_state, col] *= (remaining_prob / current_other_prob)

                stressed_matrix.loc[from_state, 'CHARGED_OFF'] = new_co_prob

    for month in range(months):
        # Check active loans (not in terminal states)
        terminal_mask = np.isin(loan_states, [state_indices['CHARGED_OFF'], state_indices['PAID_OFF']])
        active_mask = (~terminal_mask) & (balances > 0.01)

        if active_mask.sum() == 0:
            break

        # Calculate interest on active loans
        interest = np.where(active_mask, balances * monthly_rates, 0)

        # Calculate scheduled principal
        scheduled_principal = np.where(active_mask,
                                      np.minimum(monthly_payments - interest, balances),
                                      0)

        # Transition loans to new states
        new_states = loan_states.copy()
        for i in range(n_loans):
            if active_mask[i]:
                current_state = states[loan_states[i]]
                # Get transition probabilities
                trans_probs = stressed_matrix.loc[current_state].values
                # Sample new state
                new_states[i] = np.random.choice(len(states), p=trans_probs)

        # Track transitions this month
        charged_off_this_month = (new_states == state_indices['CHARGED_OFF']) & (loan_states != state_indices['CHARGED_OFF'])
        paid_off_this_month = (new_states == state_indices['PAID_OFF']) & (loan_states != state_indices['PAID_OFF'])

        # Charge-offs: lose balance, get recovery
        chargeoff_amount = np.where(charged_off_this_month, balances, 0)
        recovery_amount = chargeoff_amount * recovery_rate

        # Payoffs: receive full balance
        payoff_amount = np.where(paid_off_this_month, balances, 0)

        # Update balances
        # If charged-off or paid-off, balance goes to zero
        # Otherwise, reduce by scheduled principal
        balances = np.where(charged_off_this_month | paid_off_this_month, 0,
                           np.maximum(balances - scheduled_principal, 0))

        # Update states
        loan_states = new_states

        # Record cashflows
        total_principal = scheduled_principal.sum() + payoff_amount.sum() + recovery_amount.sum()

        cashflows.append({
            'month': month + 1,
            'interest': interest.sum(),
            'scheduled_principal': scheduled_principal.sum(),
            'prepayments': payoff_amount.sum(),
            'defaults': chargeoff_amount.sum(),
            'recoveries': recovery_amount.sum(),
            'total_inflow': interest.sum() + total_principal,
            'net_loss': chargeoff_amount.sum() - recovery_amount.sum(),
            'ending_balance': balances.sum(),
            'active_loans': active_mask.sum(),
            'current_count': (loan_states == state_indices['CURRENT']).sum(),
            'd30_count': (loan_states == state_indices['D30_59']).sum(),
            'd60_count': (loan_states == state_indices['D60_89']).sum(),
            'd90_count': (loan_states == state_indices['D90_119']).sum(),
            'chargedoff_count': (loan_states == state_indices['CHARGED_OFF']).sum(),
            'paidoff_count': (loan_states == state_indices['PAID_OFF']).sum()
        })

    return pd.DataFrame(cashflows)

# ============================================================================
# 3. RETURN CALCULATION
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
        return (total_return / initial_investment) - 1 if initial_investment > 0 else 0

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
        'irr': irr,
        'moic': moic,
        'wal_years': wal,
        'total_interest': cashflows['interest'].sum(),
        'total_losses': cashflows['net_loss'].sum(),
        'loss_rate': loss_rate
    }

# ============================================================================
# 4. RUN SCENARIOS
# ============================================================================
print("\n2. Running scenario analysis with transition model...")

# Sample portfolio
sample_size = min(10000, len(loan_tape))
np.random.seed(42)
portfolio_sample = loan_tape.dropna(subset=['approved_amount', 'int_rate', 'loan_term']).sample(
    n=sample_size, random_state=42
)

initial_value = portfolio_sample['approved_amount'].sum()
print(f"\n  Sample portfolio: {sample_size:,} loans, ${initial_value:,.0f}")

# Define scenarios
scenarios_config = {
    'Base Case': {
        'stress': 1.0,
        'recovery': 0.15
    },
    'Moderate Stress': {
        'stress': 1.5,  # 50% increase in charge-off transitions
        'recovery': 0.12
    },
    'Severe Stress': {
        'stress': 2.5,  # 150% increase in charge-off transitions
        'recovery': 0.08
    }
}

results = {}
cashflow_results = {}

for scenario_name, params in scenarios_config.items():
    print(f"\n  Running {scenario_name}...")
    print(f"    Charge-off stress: {params['stress']:.1f}x")
    print(f"    Recovery rate: {params['recovery']*100:.0f}%")

    np.random.seed(42)
    cf = project_cashflows_with_transitions(
        portfolio_sample,
        transition_matrix,
        recovery_rate=params['recovery'],
        stress_multiplier=params['stress']
    )

    cashflow_results[scenario_name] = cf

    # Calculate returns
    unlev = calculate_returns(cf, initial_value, 0.0, 0.0)
    lev = calculate_returns(cf, initial_value, 0.85, 0.065)

    results[scenario_name] = {'unlevered': unlev, 'levered': lev}

    print(f"    Unlevered IRR: {unlev['irr']*100:.2f}%")
    print(f"    Levered IRR: {lev['irr']*100:.2f}%")
    print(f"    Loss rate: {unlev['loss_rate']*100:.2f}%")

# ============================================================================
# 5. PRINT SUMMARY
# ============================================================================
print("\n" + "="*80)
print("SCENARIO ANALYSIS RESULTS (TRANSITION MODEL)")
print("="*80)

summary_data = []
for scenario_name in scenarios_config.keys():
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

# ============================================================================
# 6. SAVE RESULTS
# ============================================================================
print("\n3. Saving results...")

with open('transition_cashflow_results.pkl', 'wb') as f:
    pickle.dump({
        'results': results,
        'cashflows': cashflow_results,
        'summary': summary_df
    }, f)

print("  Results saved to transition_cashflow_results.pkl")

print("\n" + "="*80)
print("TRANSITION-BASED CASHFLOW ANALYSIS COMPLETE")
print("="*80)
