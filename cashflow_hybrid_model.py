#!/usr/bin/env python3
"""
Cashflow Projection Using Hybrid Transition Model
- Regression for Current → D1-29 and Current → Payoff
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
print(f"  Programs: {programs}, Term buckets: {len(term_buckets)}")

# Load loan data from enhanced dataset (now includes int_rate and mdr)
print("  Loading loan_performance_enhanced.csv...")
loan_perf = pd.read_csv('loan_performance_enhanced.csv')
loan_perf['report_date'] = pd.to_datetime(loan_perf['report_date'])
loan_perf['disbursement_d'] = pd.to_datetime(loan_perf['disbursement_d'])

# Get the most recent snapshot of each loan for portfolio projection
loan_tape = loan_perf.sort_values('report_date').groupby('display_id').last().reset_index()

print(f"  Loaded {len(loan_tape):,} unique loans from enhanced dataset")

# ============================================================================
# 2. HYBRID CASHFLOW PROJECTION FUNCTION
# ============================================================================

def project_cashflows_hybrid(portfolio_df, model_d1_29, scaler_d1_29, model_prepay, scaler_prepay,
                             transition_matrices, programs, term_buckets,
                             feature_cols_d1_29, feature_cols_prepay,
                             numeric_features_d1_29, numeric_features_prepay,
                             recovery_rate=0.15, months=60, stress_d1_29_mult=1.0, stress_co_mult=1.0):
    """
    Project cashflows using hybrid model:
    - Regression for Current → D1-29 (full features) and Current → Payoff (simplified features)
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

            # Build feature matrix for D1-29 model (full features)
            # Create age buckets with finer granularity in early months
            def create_age_buckets(age_months):
                if age_months <= 1:
                    return '0-1m'
                elif age_months <= 3:
                    return '2-3m'
                elif age_months <= 6:
                    return '4-6m'
                elif age_months <= 12:
                    return '7-12m'
                elif age_months <= 18:
                    return '13-18m'
                elif age_months <= 24:
                    return '19-24m'
                else:
                    return '24m+'

            age_buckets = np.array([create_age_buckets(age) for age in loan_ages[current_indices]])

            # Create FICO buckets
            def create_fico_buckets(fico):
                if fico < 620:
                    return 'fico_<620'
                elif fico < 660:
                    return 'fico_620-659'
                elif fico < 700:
                    return 'fico_660-699'
                elif fico < 740:
                    return 'fico_700-739'
                else:
                    return 'fico_740+'

            # Create amount buckets
            def create_amount_buckets(amount):
                if amount <= 2000:
                    return 'amt_0-2k'
                elif amount <= 4000:
                    return 'amt_2-4k'
                elif amount <= 6000:
                    return 'amt_4-6k'
                elif amount <= 8000:
                    return 'amt_6-8k'
                else:
                    return 'amt_8k+'

            fico_buckets = np.array([create_fico_buckets(f) for f in fico_scores[current_indices]])
            amount_buckets = np.array([create_amount_buckets(a) for a in portfolio_df['approved_amount'].values[current_indices]])
            term_vals = portfolio_df['loan_term'].values[current_indices]

            # Build feature matrix with all dummies
            d1_29_features = pd.DataFrame({
                'ever_D30': portfolio_df['ever_D30'].values[current_indices].astype(int)
            })

            # Add program dummies for D1-29 model (drop first)
            program_vals = portfolio_df['program'].values[current_indices]
            for prog in ['P2', 'P3']:
                d1_29_features[f'program_{prog}'] = (program_vals == prog).astype(int)

            # Add age bucket dummies (drop first bucket '0-1m')
            for bucket in ['2-3m', '4-6m', '7-12m', '13-18m', '19-24m', '24m+']:
                d1_29_features[f'age_{bucket}'] = (age_buckets == bucket).astype(int)

            # Add FICO bucket dummies (drop first '<620')
            for bucket in ['fico_620-659', 'fico_660-699', 'fico_700-739', 'fico_740+']:
                d1_29_features[f'{bucket}'] = (fico_buckets == bucket).astype(int)

            # Add amount bucket dummies (drop first '0-2k')
            for bucket in ['amt_2-4k', 'amt_4-6k', 'amt_6-8k', 'amt_8k+']:
                d1_29_features[f'{bucket}'] = (amount_buckets == bucket).astype(int)

            # Add loan term dummies (drop first term)
            unique_terms = sorted(set(term_vals))
            for term in unique_terms[1:]:  # Drop first term
                d1_29_features[f'term_{term}'] = (term_vals == term).astype(int)

            # Ensure all D1-29 feature columns exist
            for col in feature_cols_d1_29:
                if col not in d1_29_features.columns:
                    d1_29_features[col] = 0
            d1_29_features = d1_29_features[feature_cols_d1_29]

            # Build feature matrix for Payoff model (all categorical: program, term dummies, age buckets, FICO buckets + time to maturity)
            prepay_features = pd.DataFrame()

            # Add program dummies for Payoff model (drop first)
            for prog in ['P2', 'P3']:
                prepay_features[f'program_{prog}'] = (program_vals == prog).astype(int)

            # Add age bucket dummies (drop first bucket '0-1m')
            age_bucket_list = ['2-3m', '4-6m', '7-12m', '13-18m', '19-24m', '24m+']
            for bucket in age_bucket_list:
                prepay_features[f'age_{bucket}'] = (age_buckets == bucket).astype(int)

            # Add FICO bucket dummies (drop first '<620')
            for bucket in ['fico_620-659', 'fico_660-699', 'fico_700-739', 'fico_740+']:
                prepay_features[f'{bucket}'] = (fico_buckets == bucket).astype(int)

            # Add loan term dummies (drop first term)
            for term in unique_terms[1:]:  # Drop first term
                prepay_features[f'term_{term}'] = (term_vals == term).astype(int)

            # Add UPB bucket dummies (drop first bucket '0-1k')
            # Create UPB buckets based on current balance
            current_upbs = balances[current_indices]
            upb_buckets = np.where(current_upbs <= 1000, 'upb_0-1k',
                          np.where(current_upbs <= 2500, 'upb_1-2.5k',
                          np.where(current_upbs <= 5000, 'upb_2.5-5k',
                          np.where(current_upbs <= 7500, 'upb_5-7.5k', 'upb_7.5k+'))))

            for bucket in ['upb_1-2.5k', 'upb_2.5-5k', 'upb_5-7.5k', 'upb_7.5k+']:
                prepay_features[f'upb_{bucket}'] = (upb_buckets == bucket).astype(int)

            # Ensure all Payoff feature columns exist
            for col in feature_cols_prepay:
                if col not in prepay_features.columns:
                    prepay_features[col] = 0
            prepay_features = prepay_features[feature_cols_prepay]

            # Predict probabilities
            X_scaled_d1_29 = scaler_d1_29.transform(d1_29_features)
            prob_d1_29 = model_d1_29.predict_proba(X_scaled_d1_29)[:, 1] * stress_d1_29_mult

            X_scaled_prepay = scaler_prepay.transform(prepay_features)
            prob_prepay = model_prepay.predict_proba(X_scaled_prepay)[:, 1]

            # Determine outcomes
            rand = np.random.random(len(current_indices))

            # Payoff first (higher priority)
            prepay_mask = rand < prob_prepay
            new_states[current_indices[prepay_mask]] = 'PAID_OFF'

            # Then delinquency (for those who didn't pay off)
            remaining = ~prepay_mask
            rand2 = np.random.random(remaining.sum())
            delq_mask = rand2 < prob_d1_29[remaining]

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
            'payoffs': payoff_amount.sum(),
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
np.random.seed(42)

# Filter for 2023Q3 originations (July-September 2023) - vintage cohort analysis
# Get loans originated in 2023Q3 and still active as of Oct 2023
q3_2023_start = pd.to_datetime('2023-07-01')
q3_2023_end = pd.to_datetime('2023-09-30')
oct_2023 = pd.to_datetime('2023-10-01')

active_loans = loan_tape[
    (loan_tape['disbursement_d'] >= q3_2023_start) &
    (loan_tape['disbursement_d'] <= q3_2023_end) &
    (loan_tape['upb'] > 100) &
    (loan_tape['report_date'] >= oct_2023)
]
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


print(f"\n  Portfolio: {len(portfolio_sample):,} active loans (2023Q3 originations)")
print(f"  Origination Date Range: {portfolio_sample['disbursement_d'].min().strftime('%Y-%m-%d')} to {portfolio_sample['disbursement_d'].max().strftime('%Y-%m-%d')}")
print(f"  Current UPB: ${initial_value:,.0f}")
print(f"  Original Amount: ${portfolio_sample['approved_amount'].sum():,.0f}")

scenarios = {
    'Base Case': {'stress_d1_29': 1.0, 'stress_co': 1.0, 'recovery': 0},
    'Moderate Stress': {'stress_d1_29': 1.2, 'stress_co': 1.5, 'recovery': 0},
    'Severe Stress': {'stress_d1_29': 1.6, 'stress_co': 2.5, 'recovery': 0}
}

results = {}
cashflow_results = {}

for scenario_name, params in scenarios.items():
    print(f"\n  {scenario_name}:")
    print(f"    D1-29 stress: {params['stress_d1_29']:.1f}x, CO stress: {params['stress_co']:.1f}x, Recovery: {params['recovery']*100:.0f}%")

    np.random.seed(42)
    cf = project_cashflows_hybrid(
        portfolio_sample, model_d1_29, scaler_d1_29, model_prepay, scaler_prepay,
        transition_matrices, programs, term_buckets,
        feature_cols_d1_29, feature_cols_prepay,
        numeric_features_d1_29, numeric_features_prepay,
        recovery_rate=params['recovery'],
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
print("RESULTS SUMMARY - HYBRID TRANSITION MODEL")
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
with open('hybrid_cashflow_results.pkl', 'wb') as f:
    pickle.dump({'results': results, 'cashflows': cashflow_results, 'summary': summary_df}, f)

print("\n  Results saved to hybrid_cashflow_results.pkl")

# ============================================================================
# 6. CASHFLOW VISUALIZATION - STACKED COLUMN CHARTS
# ============================================================================
print("\n3. Creating cashflow visualizations...")

import matplotlib.pyplot as plt

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
        from matplotlib.patches import Rectangle
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

fig.suptitle('Cashflow Breakdown Comparison - All Scenarios', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('cashflow_breakdown_comparison.png', dpi=150, bbox_inches='tight')
print("  ✓ Saved cashflow_breakdown_comparison.png")
plt.close()

print("\n" + "="*80)
print("HYBRID CASHFLOW ANALYSIS COMPLETE")
print("="*80)
