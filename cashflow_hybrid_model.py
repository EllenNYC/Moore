#!/usr/bin/env python3
"""
Cashflow Projection Using Hybrid Transition Model
- Regression for Current → D30 and Current → Prepay
- Empirical Program x Term matrices for other transitions
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
programs = models['programs']
term_buckets = models['term_buckets']
feature_cols_d30 = models['feature_cols_d30']
feature_cols_prepay = models['feature_cols_prepay']
numeric_features_d30 = models['numeric_features_d30']
numeric_features_prepay = models['numeric_features_prepay']

print(f"  Loaded regression models (AUC D30: {models['auc_d30']:.3f}, Prepay: {models['auc_prepay']:.3f})")
print(f"  Loaded {len(transition_matrices)} empirical matrices (Program x Term)")
print(f"  Programs: {programs}, Term buckets: {len(term_buckets)}")

# Load loan data from enhanced dataset
print("  Loading loan_performance_enhanced.csv...")
loan_perf = pd.read_csv('loan_performance_enhanced.csv')
loan_perf['report_date'] = pd.to_datetime(loan_perf['report_date'])
loan_perf['disbursement_d'] = pd.to_datetime(loan_perf['disbursement_d'])

# Get the most recent snapshot of each loan for portfolio projection
loan_tape_enhanced = loan_perf.sort_values('report_date').groupby('display_id').last().reset_index()

# Load original loan tape for int_rate and mdr
print("  Loading original loan tape for interest rates...")
loan_tape_orig = pd.read_csv('loan tape - moore v1.0.csv')
loan_tape_orig.columns = loan_tape_orig.columns.str.strip()
loan_tape_orig['mdr'] = pd.to_numeric(loan_tape_orig['mdr'].str.rstrip('%'), errors='coerce') / 100
loan_tape_orig['int_rate'] = pd.to_numeric(loan_tape_orig['int_rate'].str.rstrip('%'), errors='coerce') / 100

# Merge enhanced data with original tape to get int_rate and mdr
loan_tape = loan_tape_enhanced.merge(
    loan_tape_orig[['display_id', 'int_rate', 'mdr']],
    on='display_id',
    how='left'
)

print(f"  Loaded {len(loan_tape):,} unique loans from enhanced dataset")

# ============================================================================
# 2. HYBRID CASHFLOW PROJECTION FUNCTION
# ============================================================================

def project_cashflows_hybrid(portfolio_df, model_d30, scaler_d30, model_prepay, scaler_prepay,
                             transition_matrices, programs, term_buckets,
                             feature_cols_d30, feature_cols_prepay,
                             numeric_features_d30, numeric_features_prepay,
                             recovery_rate=0.15, months=60, stress_d30_mult=1.0, stress_co_mult=1.0):
    """
    Project cashflows using hybrid model:
    - Regression for Current → D30 (full features) and Current → Prepay (simplified features)
    - Empirical Program x Term matrices for delinquency transitions
    """

    n_loans = len(portfolio_df)
    cashflows = []

    # Initialize - use current UPB as starting balance
    balances = portfolio_df['upb'].values.copy()
    monthly_rates = portfolio_df['int_rate'].values / 12
    terms = portfolio_df['loan_term'].values
    fico_scores = portfolio_df['fico_score'].values
    original_amounts = portfolio_df['approved_amount'].values

    # Calculate monthly payments based on original amount
    monthly_payments = np.zeros(n_loans)
    for i in range(n_loans):
        if monthly_rates[i] > 0 and terms[i] > 0:
            r = monthly_rates[i]
            n = terms[i]
            # Payment based on original amount
            monthly_payments[i] = original_amounts[i] * (r * (1 + r)**n) / ((1 + r)**n - 1)
        else:
            monthly_payments[i] = balances[i] / max(terms[i], 1)

    # Initialize states from current delinquency status
    # Map delinquency_bucket to state
    state_mapping = {
        'CURRENT': 'CURRENT',
        '1-30 DPD': 'D1_29',
        '31-60 DPD': 'D30_59',
        '61-90 DPD': 'D60_89',
        '91-120 DPD': 'D90_119',
        '120+ DPD': 'D120_PLUS',
        'PAID OFF': 'PAID_OFF',
        'CHARGED OFF': 'CHARGED_OFF',
        'SATISFIED': 'PAID_OFF',
        'WRITTEN OFF': 'CHARGED_OFF'
    }

    states = portfolio_df['delinquency_bucket'].map(state_mapping).fillna('CURRENT').values

    # Use current loan age
    loan_ages = portfolio_df['loan_age_months'].values.copy()

    # Program and Term buckets for each loan
    loan_programs = portfolio_df['program'].values

    # Categorize terms into buckets
    def categorize_term(term):
        if term <= 3:
            return '0-3m'
        elif term <= 6:
            return '4-6m'
        elif term <= 12:
            return '7-12m'
        elif term <= 18:
            return '13-18m'
        elif term <= 24:
            return '19-24m'
        else:
            return '24m+'

    term_bucket_vals = np.array([categorize_term(t) for t in terms])

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

        # Transition loans
        new_states = states.copy()

        # ====================================================================
        # CURRENT LOANS: Use Regression Models
        # ====================================================================
        current_mask = (states == 'CURRENT') & active_mask

        if current_mask.sum() > 0:
            # Prepare features for regression
            current_indices = np.where(current_mask)[0]

            # Build feature matrix for D30+ model (full features)
            d30_features = pd.DataFrame({
                'fico_score': fico_scores[current_indices],
                'approved_amount': portfolio_df['approved_amount'].values[current_indices],
                'loan_term': portfolio_df['loan_term'].values[current_indices],
                'loan_age_months': loan_ages[current_indices],
                'upb': balances[current_indices],
                'paid_principal': portfolio_df['paid_principal'].values[current_indices],
                'paid_interest': portfolio_df['paid_interest'].values[current_indices],
                'ever_D30': portfolio_df['ever_D30'].values[current_indices],
                'ever_D60': portfolio_df['ever_D60'].values[current_indices],
                'ever_D90': portfolio_df['ever_D90'].values[current_indices]
            })

            # Add program dummies for D30+ model
            program_vals = portfolio_df['program'].values[current_indices]
            for prog in ['P2', 'P3']:
                d30_features[f'program_{prog}'] = (program_vals == prog).astype(int)

            # Ensure all D30+ feature columns exist
            for col in feature_cols_d30:
                if col not in d30_features.columns:
                    d30_features[col] = 0
            d30_features = d30_features[feature_cols_d30]

            # Build feature matrix for Prepay model (simplified: program, term, age only)
            prepay_features = pd.DataFrame({
                'loan_term': portfolio_df['loan_term'].values[current_indices],
                'loan_age_months': loan_ages[current_indices]
            })

            # Add program dummies for Prepay model
            for prog in ['P2', 'P3']:
                prepay_features[f'program_{prog}'] = (program_vals == prog).astype(int)

            # Ensure all Prepay feature columns exist
            for col in feature_cols_prepay:
                if col not in prepay_features.columns:
                    prepay_features[col] = 0
            prepay_features = prepay_features[feature_cols_prepay]

            # Predict probabilities
            X_scaled_d30 = scaler_d30.transform(d30_features)
            prob_d30 = model_d30.predict_proba(X_scaled_d30)[:, 1] * stress_d30_mult

            X_scaled_prepay = scaler_prepay.transform(prepay_features)
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
                program = loan_programs[idx]
                term_bucket = term_bucket_vals[idx]

                # Lookup transition probabilities
                matrix_key = (program, term_bucket)

                if matrix_key in trans_matrix:
                    trans_probs = trans_matrix[matrix_key]
                else:
                    # Fallback to Program-only (average across all terms for this program)
                    fallback_keys = [k for k in trans_matrix.keys() if k[0] == program]
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
    # print(pd.DataFrame(cashflows))

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

# Filter for active loans only (exclude already charged off or paid off)
active_loans = loan_tape[~loan_tape['delinquency_bucket'].isin(['PAID OFF', 'CHARGED OFF', 'SATISFIED', 'WRITTEN OFF'])]
required_cols = ['approved_amount', 'int_rate', 'loan_term', 'fico_score', 'program',
                 'upb', 'paid_principal', 'paid_interest', 'ever_D30', 'ever_D60', 'ever_D90',
                 'delinquency_bucket', 'loan_age_months']

# Check for delinquency_bucket column
if 'delinquency_bucket' not in active_loans.columns:
    print("  Warning: delinquency_bucket not found, using loan_status instead")
    # Map loan_status to delinquency_bucket
    active_loans = loan_tape[~loan_tape['loan_status'].isin(['PAID OFF', 'CHARGED OFF', 'SATISFIED', 'WRITTEN OFF'])]

portfolio_sample = active_loans.dropna(subset=required_cols).sample(
    n=min(sample_size, len(active_loans)), random_state=42
)

# Use current UPB as initial value (not original approved amount)
# assume we pay 1% upfront fees
price_percent = 1-portfolio_sample['mdr']+0.01
initial_value = (portfolio_sample['upb'] * price_percent).sum()


print(f"\n  Portfolio: {len(portfolio_sample):,} active loans")
print(f"  Current UPB: ${initial_value:,.0f}")
print(f"  Original Amount: ${portfolio_sample['approved_amount'].sum():,.0f}")

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
        transition_matrices, programs, term_buckets,
        feature_cols_d30, feature_cols_prepay,
        numeric_features_d30, numeric_features_prepay,
        recovery_rate=params['recovery'],
        stress_d30_mult=params['stress_d30'],
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
