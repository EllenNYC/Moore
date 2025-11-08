#!/usr/bin/env python3
"""
Cashflow Projection Using Hybrid Transition Model
- Regression for Current → D30 and Current → Prepay
- Empirical FICO x Age matrices for other transitions
"""

import pandas as pd
import numpy as np
import pickle
from scipy import optimize
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("CASHFLOW PROJECTION - HYBRID TRANSITION MODEL")
print("="*80)

# ============================================================================
# 1. LOAD MODELS AND DATA
# ============================================================================
print("\n1. Loading hybrid transition models...")

with open('hybrid_transition_models.pkl', 'rb') as f:
    models = pickle.load(f)

model_d30 = models['model_d30']
scaler_d30 = models['scaler_d30']
model_prepay = models['model_prepay']
scaler_prepay = models['scaler_prepay']
transition_matrices = models['transition_matrices']
fico_bins = models['fico_bins']
fico_labels = models['fico_labels']
age_bins = models['age_bins']
age_labels = models['age_labels']
feature_cols = models['feature_cols']
numeric_features = models['numeric_features']

print(f"  Loaded regression models (AUC D30: {models['auc_d30']:.3f}, Prepay: {models['auc_prepay']:.3f})")
print(f"  Loaded {len(transition_matrices)} empirical matrices")

# Load loan data
loan_tape = pd.read_csv('loan tape - moore v1.0.csv')
loan_tape.columns = loan_tape.columns.str.strip()
loan_tape['mdr'] = pd.to_numeric(loan_tape['mdr'].str.rstrip('%'), errors='coerce') / 100
loan_tape['int_rate'] = pd.to_numeric(loan_tape['int_rate'].str.rstrip('%'), errors='coerce') / 100
loan_tape['approved_amount'] = pd.to_numeric(loan_tape['approved_amount'].str.replace('$', '').str.replace(',', ''), errors='coerce')

# ============================================================================
# 2. HYBRID CASHFLOW PROJECTION FUNCTION
# ============================================================================

def project_cashflows_hybrid(portfolio_df, model_d30, scaler_d30, model_prepay, scaler_prepay,
                             transition_matrices, fico_bins, fico_labels, age_bins, age_labels,
                             feature_cols, numeric_features,
                             recovery_rate=0.15, months=60, stress_d30_mult=1.0, stress_co_mult=1.0):
    """
    Project cashflows using hybrid model:
    - Regression for Current → D30 and Current → Prepay
    - Empirical matrices for delinquency transitions
    """

    n_loans = len(portfolio_df)
    cashflows = []

    # Initialize
    balances = portfolio_df['approved_amount'].values.copy()
    monthly_rates = portfolio_df['int_rate'].values / 12
    terms = portfolio_df['loan_term'].values
    fico_scores = portfolio_df['fico_score'].values

    # Calculate monthly payments
    monthly_payments = np.zeros(n_loans)
    for i in range(n_loans):
        if monthly_rates[i] > 0 and terms[i] > 0:
            r = monthly_rates[i]
            n = terms[i]
            monthly_payments[i] = balances[i] * (r * (1 + r)**n) / ((1 + r)**n - 1)
        else:
            monthly_payments[i] = balances[i] / max(terms[i], 1)

    # All loans start CURRENT
    states = np.array(['CURRENT'] * n_loans)
    loan_ages = np.zeros(n_loans)  # Age in months

    # FICO and Age buckets for each loan
    fico_buckets = pd.cut(fico_scores, bins=fico_bins, labels=fico_labels)
    fico_buckets = fico_buckets.astype(str)

    for month in range(months):
        # Active loans (not charged-off or paid-off)
        active_mask = ~np.isin(states, ['CHARGED_OFF', 'PAID_OFF']) & (balances > 0.01)

        if active_mask.sum() == 0:
            break

        # Calculate interest
        interest = np.where(active_mask, balances * monthly_rates, 0)

        # Scheduled principal
        scheduled_principal = np.where(active_mask,
                                      np.minimum(monthly_payments - interest, balances),
                                      0)

        # Determine age buckets
        age_buckets = pd.cut(loan_ages, bins=age_bins, labels=age_labels, right=False)
        age_buckets = age_buckets.astype(str)

        # Transition loans
        new_states = states.copy()

        # ====================================================================
        # CURRENT LOANS: Use Regression Models
        # ====================================================================
        current_mask = (states == 'CURRENT') & active_mask

        if current_mask.sum() > 0:
            # Prepare features for regression
            current_indices = np.where(current_mask)[0]

            # Build feature matrix
            current_features = pd.DataFrame({
                'fico_score': fico_scores[current_indices],
                'approved_amount': portfolio_df['approved_amount'].values[current_indices],
                'loan_term': portfolio_df['loan_term'].values[current_indices],
                'int_rate': portfolio_df['int_rate'].values[current_indices],
                'mdr': portfolio_df['mdr'].values[current_indices],
                'loan_age_months': loan_ages[current_indices]
            })

            # Add program dummies
            program_vals = portfolio_df['program'].values[current_indices]
            for prog in ['P2', 'P3']:
                current_features[f'program_{prog}'] = (program_vals == prog).astype(int)

            # Ensure all feature columns exist
            for col in feature_cols:
                if col not in current_features.columns:
                    current_features[col] = 0
            current_features = current_features[feature_cols]

            # Predict probabilities
            X_scaled_d30 = scaler_d30.transform(current_features)
            prob_d30 = model_d30.predict_proba(X_scaled_d30)[:, 1] * stress_d30_mult

            X_scaled_prepay = scaler_prepay.transform(current_features)
            prob_prepay = model_prepay.predict_proba(X_scaled_prepay)[:, 1]

            # Determine outcomes
            rand = np.random.random(len(current_indices))

            # Prepay first (higher priority)
            prepay_mask = rand < prob_prepay
            new_states[current_indices[prepay_mask]] = 'PAID_OFF'

            # Then delinquency (for those who didn't prepay)
            remaining = ~prepay_mask
            rand2 = np.random.random(remaining.sum())
            delq_mask = rand2 < prob_d30[remaining]

            remaining_indices = current_indices[remaining][delq_mask]
            new_states[remaining_indices] = 'D1_29'  # First bucket

        # ====================================================================
        # DELINQUENT LOANS: Use Empirical Matrices
        # ====================================================================
        delinq_states = ['D1_29', 'D30_59', 'D60_89', 'D90_119', 'D120_PLUS']

        for delq_state in delinq_states:
            state_mask = (states == delq_state) & active_mask

            if state_mask.sum() == 0:
                continue

            if delq_state not in transition_matrices:
                continue  # Skip if no matrix

            state_indices = np.where(state_mask)[0]
            trans_matrix = transition_matrices[delq_state]

            for idx in state_indices:
                fico_bucket = fico_buckets[idx]
                age_bucket = age_buckets[idx]

                # Lookup transition probabilities
                matrix_key = (fico_bucket, age_bucket)

                if matrix_key in trans_matrix:
                    trans_probs = trans_matrix[matrix_key]
                else:
                    # Fallback to FICO-only
                    fallback_keys = [k for k in trans_matrix.keys() if k[0] == fico_bucket]
                    if fallback_keys:
                        trans_probs = trans_matrix[fallback_keys[0]]
                    else:
                        # Ultimate fallback: first matrix entry
                        trans_probs = list(trans_matrix.values())[0]

                # Apply stress to charge-offs
                co_prob = trans_probs.get('CHARGED_OFF', 0) * stress_co_mult
                co_prob = min(co_prob, 0.99)

                # Sample transition
                all_states_list = ['CURRENT', 'D1_29', 'D30_59', 'D60_89', 'D90_119',
                                   'D120_PLUS', 'CHARGED_OFF', 'PAID_OFF']
                probs = [trans_probs.get(s, 0) for s in all_states_list]

                # Adjust for stressed charge-off
                if stress_co_mult != 1.0:
                    orig_co = trans_probs.get('CHARGED_OFF', 0)
                    prob_increase = co_prob - orig_co
                    # Reduce other probabilities proportionally
                    total_other = sum(probs) - orig_co
                    if total_other > 0:
                        for i, s in enumerate(all_states_list):
                            if s != 'CHARGED_OFF':
                                probs[i] = max(0, probs[i] - (probs[i] / total_other) * prob_increase)
                    probs[all_states_list.index('CHARGED_OFF')] = co_prob

                # Normalize
                prob_sum = sum(probs)
                if prob_sum > 0:
                    probs = [p / prob_sum for p in probs]
                else:
                    probs = [1.0 if s == delq_state else 0.0 for s in all_states_list]

                # Sample new state
                new_state = np.random.choice(all_states_list, p=probs)
                new_states[idx] = new_state

        # Track transitions this month
        charged_off = (new_states == 'CHARGED_OFF') & (states != 'CHARGED_OFF')
        paid_off = (new_states == 'PAID_OFF') & (states != 'PAID_OFF')

        chargeoff_amount = np.where(charged_off, balances, 0)
        recovery_amount = chargeoff_amount * recovery_rate

        payoff_amount = np.where(paid_off, balances, 0)

        # Update balances
        balances = np.where(charged_off | paid_off, 0,
                           np.maximum(balances - scheduled_principal, 0))

        # Update states and ages
        states = new_states
        loan_ages += 1

        # Record cashflows
        cashflows.append({
            'month': month + 1,
            'interest': interest.sum(),
            'scheduled_principal': scheduled_principal.sum(),
            'prepayments': payoff_amount.sum(),
            'defaults': chargeoff_amount.sum(),
            'recoveries': recovery_amount.sum(),
            'total_inflow': interest.sum() + scheduled_principal.sum() + payoff_amount.sum() + recovery_amount.sum(),
            'net_loss': chargeoff_amount.sum() - recovery_amount.sum(),
            'ending_balance': balances.sum(),
            'active_loans': active_mask.sum(),
            'current_count': (states == 'CURRENT').sum(),
            'chargedoff_count': (states == 'CHARGED_OFF').sum(),
            'paidoff_count': (states == 'PAID_OFF').sum()
        })

    return pd.DataFrame(cashflows)

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
np.random.seed(42)
portfolio_sample = loan_tape.dropna(subset=['approved_amount', 'int_rate', 'loan_term', 'fico_score', 'mdr', 'program']).sample(
    n=sample_size, random_state=42
)

initial_value = portfolio_sample['approved_amount'].sum()
print(f"\n  Portfolio: {sample_size:,} loans, ${initial_value:,.0f}")

scenarios = {
    'Base Case': {'stress_d30': 1.0, 'stress_co': 1.0, 'recovery': 0.15},
    'Moderate Stress': {'stress_d30': 1.3, 'stress_co': 1.5, 'recovery': 0.12},
    'Severe Stress': {'stress_d30': 1.6, 'stress_co': 2.5, 'recovery': 0.08}
}

results = {}
cashflow_results = {}

for scenario_name, params in scenarios.items():
    print(f"\n  {scenario_name}:")
    print(f"    D30 stress: {params['stress_d30']:.1f}x, CO stress: {params['stress_co']:.1f}x, Recovery: {params['recovery']*100:.0f}%")

    np.random.seed(42)
    cf = project_cashflows_hybrid(
        portfolio_sample, model_d30, scaler_d30, model_prepay, scaler_prepay,
        transition_matrices, fico_bins, fico_labels, age_bins, age_labels,
        feature_cols, numeric_features,
        recovery_rate=params['recovery'],
        stress_d30_mult=params['stress_d30'],
        stress_co_mult=params['stress_co']
    )

    cashflow_results[scenario_name] = cf

    unlev = calculate_returns(cf, initial_value, 0.0, 0.0)
    lev = calculate_returns(cf, initial_value, 0.85, 0.065)

    results[scenario_name] = {'unlevered': unlev, 'levered': lev}

    print(f"    Unlevered IRR: {unlev['irr']*100:.2f}%, Loss: {unlev['loss_rate']*100:.1f}%")
    print(f"    Levered IRR: {lev['irr']*100:.2f}%")

# ============================================================================
# 5. SUMMARY
# ============================================================================
print("\n" + "="*80)
print("RESULTS SUMMARY - HYBRID TRANSITION MODEL")
print("="*80)

summary_data = []
for scenario_name in scenarios.keys():
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

# Save
with open('hybrid_cashflow_results.pkl', 'wb') as f:
    pickle.dump({'results': results, 'cashflows': cashflow_results, 'summary': summary_df}, f)

print("\n  Results saved to hybrid_cashflow_results.pkl")

print("\n" + "="*80)
print("HYBRID CASHFLOW ANALYSIS COMPLETE")
print("="*80)
