#!/usr/bin/env python3
"""
Hybrid Transition Model for Consumer Credit Portfolio
- Uses loan_performance_enhanced.csv with pre-computed features
- Regression for Current → D30 and Current → Payoff
- Empirical matrices (Program x Term) for all other transitions
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("HYBRID TRANSITION MODEL - REGRESSION + EMPIRICAL MATRICES")
print("Using Enhanced Dataset: loan_performance_enhanced.csv")
print("="*80)

# ============================================================================
# 1. LOAD AND PREPARE DATA
# ============================================================================
print("\n1. Loading enhanced dataset...")

# Load enhanced dataset (already has loan tape merged in)
df = pd.read_csv('loan_performance_enhanced.csv')

# Parse dates
df['report_date'] = pd.to_datetime(df['report_date'])
df['disbursement_d'] = pd.to_datetime(df['disbursement_d'])

print(f"  Loaded {df['display_id'].nunique():,} unique loans")
print(f"  Total observations: {len(df):,}")
print(f"  Date range: {df['report_date'].min()} to {df['report_date'].max()}")

# ============================================================================
# 2. CREATE STATES AND TRANSITIONS
# ============================================================================
print("\n2. Creating delinquency states and transitions...")

# Define delinquency states (using delinquency_bucket already in data)
# Map to standardized state names
state_mapping = {
    'CURRENT': 'CURRENT',
    '1-29 DPD': 'D1_29',
    '30-59 DPD': 'D30_59',
    '60-89 DPD': 'D60_89',
    '90-119 DPD': 'D90_119',
    '120+ DPD': 'D120_PLUS',
    'Paid_off': 'PAID_OFF',
    'Default': 'CHARGED_OFF'
}

# Assign state from delinquency_bucket
df['state'] = df['delinquency_bucket'].map(state_mapping)

# Fill any unmapped values with CURRENT as default
df['state'] = df['state'].fillna('CURRENT')

print(f"  State distribution:")
print(df['state'].value_counts())

# Sort by loan and date
df = df.sort_values(['display_id', 'report_date'])

# Create lagged state (previous month's state)
df['prev_state'] = df.groupby('display_id')['state'].shift(1)

# Exclude first observation (no previous state)
transitions = df[df['prev_state'].notna()].copy()

# Exclude transitions FROM terminal states (no transitions after charge-off or paid-off)
transitions = transitions[~transitions['prev_state'].isin(['CHARGED_OFF', 'PAID_OFF'])].copy()

print(f"  Total valid transitions: {len(transitions):,}")
print(f"  Transition date range: {transitions['report_date'].min()} to {transitions['report_date'].max()}")

# ============================================================================
# 3. BUILD REGRESSION MODELS (CURRENT → D30 AND CURRENT → PAYOFF)
# ============================================================================
print("\n3. Building regression models for Current state transitions...")

# Filter to transitions from CURRENT state only
current_transitions = transitions[transitions['prev_state'] == 'CURRENT'].copy()

print(f"  Current-state transitions (before filtering): {len(current_transitions):,}")

# IMPORTANT: Exclude transitions at or after loan maturity
# Maturity occurs when loan_age_months >= loan_term
# We only want to model behavior BEFORE the loan matures
print(f"\n  Filtering out transitions at or after maturity (loan_age >= loan_term)...")
before_filter = len(current_transitions)
current_transitions = current_transitions[current_transitions['loan_age_months'] < current_transitions['loan_term']].copy()
after_filter = len(current_transitions)
print(f"  Removed {before_filter - after_filter:,} transitions at/after maturity ({(before_filter - after_filter)/before_filter*100:.1f}%)")

print(f"  Current-state transitions (after filtering): {len(current_transitions):,}")
print(f"  Next state distribution:")
print(current_transitions['state'].value_counts())

# Create outcome variables
# Change to predict Current → D1-29 (early delinquency) instead of D30+
current_transitions['to_d1_29'] = (current_transitions['state'] == 'D1_29').astype(int)
# Prepay = PAID_OFF, but only if it's BEFORE maturity (which is already filtered above)
current_transitions['to_prepay'] = (current_transitions['state'] == 'PAID_OFF').astype(int)
current_transitions['to_chargeoff'] = (current_transitions['state'] == 'CHARGED_OFF').astype(int)

print(f"\n  Outcome rates (before maturity only):")
print(f"    → D1-29 (Early Delinquency): {current_transitions['to_d1_29'].mean()*100:.2f}%")
print(f"    → Payoff (Early Payoff): {current_transitions['to_prepay'].mean()*100:.2f}%")
print(f"    → Direct charge-off: {current_transitions['to_chargeoff'].mean()*100:.2f}%")

# Enhanced features from the dataset
# Create loan age buckets for D1-29 model
def create_age_buckets(age_months):
    """Create loan age buckets with finer granularity in early months"""
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

current_transitions['age_bucket'] = current_transitions['loan_age_months'].apply(create_age_buckets)

# Create FICO score buckets
def create_fico_buckets(fico):
    """Create FICO score buckets"""
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

# Create loan amount buckets
def create_amount_buckets(amount):
    """Create loan amount buckets"""
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

# Create UPB buckets (for prepay model)
def create_upb_buckets(upb):
    """Create UPB (unpaid principal balance) buckets"""
    if upb <= 1000:
        return 'upb_0-1k'
    elif upb <= 2500:
        return 'upb_1-2.5k'
    elif upb <= 5000:
        return 'upb_2.5-5k'
    elif upb <= 7500:
        return 'upb_5-7.5k'
    else:
        return 'upb_7.5k+'

current_transitions['fico_bucket'] = current_transitions['fico_score'].apply(create_fico_buckets)
current_transitions['amount_bucket'] = current_transitions['approved_amount'].apply(create_amount_buckets)
current_transitions['upb_bucket'] = current_transitions['upb'].apply(create_upb_buckets)

# No numeric features - all categorical
# Add program dummies
program_dummies = pd.get_dummies(current_transitions['program'], prefix='program', drop_first=True)

# Add age bucket dummies (drop first to avoid multicollinearity)
age_bucket_dummies = pd.get_dummies(current_transitions['age_bucket'], prefix='age', drop_first=True)

# Add FICO bucket dummies (drop first)
fico_bucket_dummies = pd.get_dummies(current_transitions['fico_bucket'], prefix='fico', drop_first=True)

# Add amount bucket dummies (drop first)
amount_bucket_dummies = pd.get_dummies(current_transitions['amount_bucket'], prefix='amt', drop_first=True)

# Add loan term dummies (drop first)
loan_term_dummies = pd.get_dummies(current_transitions['loan_term'], prefix='term', drop_first=True)

# Add ever_D30 dummy (binary, no need to drop)
ever_d30_dummy = current_transitions[['ever_D30']].astype(int)

# Combine all dummy features
X = pd.concat([program_dummies, age_bucket_dummies, fico_bucket_dummies,
               amount_bucket_dummies, loan_term_dummies, ever_d30_dummy], axis=1)

# ============================================================================
# Model A: Current → D1-29 (Early Delinquency)
# ============================================================================
print("\n  Model A: Current → D1-29 (Early Delinquency)...")

y_d1_29 = current_transitions['to_d1_29']
valid_idx = X.notna().all(axis=1) & y_d1_29.notna()
X_clean = X[valid_idx]
y_clean = y_d1_29[valid_idx]

print(f"    Valid observations: {len(X_clean):,}")
print(f"    Positive class rate: {y_clean.mean()*100:.2f}%")

# Get indices for train/test split to track report dates
train_idx, test_idx = train_test_split(
    X_clean.index, test_size=0.3, random_state=42,
    stratify=y_clean
)

X_train = X_clean.loc[train_idx]
X_test = X_clean.loc[test_idx]
y_train = y_clean.loc[train_idx]
y_test = y_clean.loc[test_idx]

scaler_d1_29 = StandardScaler()
X_train_scaled = scaler_d1_29.fit_transform(X_train)
X_test_scaled = scaler_d1_29.transform(X_test)

model_d1_29 = LogisticRegression(random_state=42, max_iter=1000)
model_d1_29.fit(X_train_scaled, y_train)

# Evaluate
y_train_pred_proba = model_d1_29.predict_proba(X_train_scaled)[:, 1]
y_test_pred_proba = model_d1_29.predict_proba(X_test_scaled)[:, 1]
y_pred = model_d1_29.predict(X_test_scaled)
auc_d1_29 = roc_auc_score(y_test, y_test_pred_proba)

print(f"    AUC-ROC: {auc_d1_29:.4f}")
print(f"    Test accuracy: {(y_pred == y_test).mean()*100:.2f}%")

# Feature importance
feature_cols = list(X_clean.columns)
coefs_d1_29 = pd.DataFrame({
    'Feature': feature_cols,
    'Coefficient': model_d1_29.coef_[0]
}).sort_values('Coefficient', key=abs, ascending=False)

print(f"\n    Top 5 features (by absolute coefficient):")
for idx, row in coefs_d1_29.head(5).iterrows():
    print(f"      {row['Feature']:20s}: {row['Coefficient']:+.4f}")

# Create prediction vs actual by loan age (D1-29 model)
d1_29_train_results = pd.DataFrame({
    'loan_age_months': current_transitions.loc[train_idx, 'loan_age_months'],
    'actual': y_train,
    'predicted_prob': y_train_pred_proba,
    'sample': 'train',
    'model': 'Current_to_D1_29'
})

d1_29_test_results = pd.DataFrame({
    'loan_age_months': current_transitions.loc[test_idx, 'loan_age_months'],
    'actual': y_test,
    'predicted_prob': y_test_pred_proba,
    'sample': 'test',
    'model': 'Current_to_D1_29'
})

d1_29_results_combined = pd.concat([d1_29_train_results, d1_29_test_results], ignore_index=True)

# Aggregate by loan_age_months and sample
d1_29_by_age = d1_29_results_combined.groupby(['loan_age_months', 'sample']).agg({
    'actual': ['mean', 'count'],
    'predicted_prob': 'mean'
}).reset_index()

d1_29_by_age.columns = ['loan_age_months', 'sample', 'actual_rate', 'num_obs', 'predicted_rate']
d1_29_by_age['model'] = 'Current_to_D1_29'
# Filter to only show age >= 1 month
d1_29_by_age = d1_29_by_age[d1_29_by_age['loan_age_months'] >= 1]
d1_29_by_age = d1_29_by_age.sort_values(['loan_age_months', 'sample'])

print(f"\n    Prediction vs Actual by Loan Age (D1-29)")

# ============================================================================
# Model B: Current → Payoff
# ============================================================================
print("\n  Model B: Current → Payoff...")
print("    Using all categorical features: program, loan_term (dummies), age buckets, FICO buckets, UPB buckets")

# Build prepay feature matrix with all dummies
# Add program dummies
prepay_program_dummies = pd.get_dummies(current_transitions['program'], prefix='program', drop_first=True)

# Add age bucket dummies (drop first to avoid multicollinearity)
# Age buckets already created above for D1-29 model
prepay_age_bucket_dummies = pd.get_dummies(current_transitions['age_bucket'], prefix='age', drop_first=True)

# Add FICO bucket dummies (drop first) - FICO buckets already created above
prepay_fico_bucket_dummies = pd.get_dummies(current_transitions['fico_bucket'], prefix='fico', drop_first=True)

# Add UPB bucket dummies (drop second bucket '1-2.5k' instead of first) - NEW FEATURE replacing approved_amount
all_upb_dummies = pd.get_dummies(current_transitions['upb_bucket'], prefix='upb')
# Keep all buckets except the second one (upb_1-2.5k)
prepay_upb_bucket_dummies = all_upb_dummies.drop(columns=['upb_upb_1-2.5k'])

# Add loan term dummies (drop first)
prepay_loan_term_dummies = pd.get_dummies(current_transitions['loan_term'], prefix='term', drop_first=True)

# Combine all dummy features for prepay model (removed time_to_maturity since UPB captures that signal)
X_prepay = pd.concat([prepay_program_dummies, prepay_age_bucket_dummies, prepay_fico_bucket_dummies,
                      prepay_upb_bucket_dummies, prepay_loan_term_dummies], axis=1)

y_prepay = current_transitions['to_prepay']
valid_idx = X_prepay.notna().all(axis=1) & y_prepay.notna()
X_prepay_clean = X_prepay[valid_idx]
y_clean = y_prepay[valid_idx]

print(f"    Valid observations: {len(X_prepay_clean):,}")
print(f"    Positive class rate: {y_clean.mean()*100:.2f}%")
print(f"    Features: {len(X_prepay_clean.columns)} ({len(prepay_program_dummies.columns)} program + {len(prepay_age_bucket_dummies.columns)} age + {len(prepay_fico_bucket_dummies.columns)} FICO + {len(prepay_upb_bucket_dummies.columns)} UPB + {len(prepay_loan_term_dummies.columns)} term)")

# Get indices for train/test split to track report dates
prepay_train_idx, prepay_test_idx = train_test_split(
    X_prepay_clean.index, test_size=0.3, random_state=42,
    stratify=y_clean
)

X_train = X_prepay_clean.loc[prepay_train_idx]
X_test = X_prepay_clean.loc[prepay_test_idx]
y_train = y_clean.loc[prepay_train_idx]
y_test = y_clean.loc[prepay_test_idx]

scaler_prepay = StandardScaler()
X_train_scaled = scaler_prepay.fit_transform(X_train)
X_test_scaled = scaler_prepay.transform(X_test)

model_prepay = LogisticRegression(random_state=42, max_iter=1000)
model_prepay.fit(X_train_scaled, y_train)

# Evaluate
y_train_pred_proba_prepay = model_prepay.predict_proba(X_train_scaled)[:, 1]
y_test_pred_proba_prepay = model_prepay.predict_proba(X_test_scaled)[:, 1]
y_pred = model_prepay.predict(X_test_scaled)
auc_prepay = roc_auc_score(y_test, y_test_pred_proba_prepay)

print(f"    AUC-ROC: {auc_prepay:.4f}")
print(f"    Test accuracy: {(y_pred == y_test).mean()*100:.2f}%")

prepay_feature_cols = list(X_prepay_clean.columns)
coefs_prepay = pd.DataFrame({
    'Feature': prepay_feature_cols,
    'Coefficient': model_prepay.coef_[0]
}).sort_values('Coefficient', key=abs, ascending=False)

print(f"\n    Top 5 features (by absolute coefficient):")
for idx, row in coefs_prepay.head(5).iterrows():
    print(f"      {row['Feature']:20s}: {row['Coefficient']:+.4f}")

# Create prediction vs actual by report date (Prepay model)
prepay_train_results = pd.DataFrame({
    'loan_age_months': current_transitions.loc[prepay_train_idx, 'loan_age_months'],
    'actual': y_train,
    'predicted_prob': y_train_pred_proba_prepay,
    'sample': 'train',
    'model': 'Current_to_Prepay'
})

prepay_test_results = pd.DataFrame({
    'loan_age_months': current_transitions.loc[prepay_test_idx, 'loan_age_months'],
    'actual': y_test,
    'predicted_prob': y_test_pred_proba_prepay,
    'sample': 'test',
    'model': 'Current_to_Prepay'
})

prepay_results_combined = pd.concat([prepay_train_results, prepay_test_results], ignore_index=True)

# Aggregate by loan_age_months and sample
prepay_by_age = prepay_results_combined.groupby(['loan_age_months', 'sample']).agg({
    'actual': ['mean', 'count'],
    'predicted_prob': 'mean'
}).reset_index()

prepay_by_age.columns = ['loan_age_months', 'sample', 'actual_rate', 'num_obs', 'predicted_rate']
prepay_by_age['model'] = 'Current_to_Prepay'
# Filter to only show age >= 1 month
prepay_by_age = prepay_by_age[prepay_by_age['loan_age_months'] >= 1]
prepay_by_age = prepay_by_age.sort_values(['loan_age_months', 'sample'])

print(f"\n    Prediction vs Actual by Loan Age (Payoff)")

# ============================================================================
# 4. BUILD EMPIRICAL MATRICES (PROGRAM x TERM) FOR OTHER TRANSITIONS
# ============================================================================
print("\n4. Building empirical transition matrices (Program x Loan Term)...")

# Get unique programs
programs = sorted(transitions['program'].unique())
print(f"  Programs: {programs}")

# Define term buckets
transitions['term_bucket'] = transitions['loan_term']
term_buckets = transitions['term_bucket'].unique().tolist()

# Define transition states
delinq_states = ['D1_29', 'D30_59', 'D60_89', 'D90_119', 'D120_PLUS']
all_states = ['CURRENT'] + delinq_states + ['CHARGED_OFF', 'PAID_OFF']

# Build matrices for each delinquency state
transition_matrices = {}

for from_state in delinq_states:
    print(f"\n  Building matrix for {from_state}...")

    state_trans = transitions[transitions['prev_state'] == from_state].copy()

    if len(state_trans) < 100:
        print(f"    Insufficient data ({len(state_trans)} obs), skipping")
        continue

    print(f"    Total transitions: {len(state_trans):,}")

    # Create matrix: rows = Programs, columns = Term buckets
    matrix_dict = {}

    for program in programs:
        for term in term_buckets:
            subset = state_trans[(state_trans['program'] == program) &
                                 (state_trans['term_bucket'] == term)]

            if len(subset) >= 10:  # Minimum observations
                # Calculate transition probabilities
                trans_probs = {}
                for to_state in all_states:
                    trans_probs[to_state] = (subset['state'] == to_state).mean()

                matrix_dict[(program, term)] = trans_probs
            else:
                # Fall back to program average
                program_subset = state_trans[state_trans['program'] == program]
                if len(program_subset) >= 10:
                    trans_probs = {}
                    for to_state in all_states:
                        trans_probs[to_state] = (program_subset['state'] == to_state).mean()
                    matrix_dict[(program, term)] = trans_probs
                else:
                    # Fall back to overall average for this state
                    trans_probs = {}
                    for to_state in all_states:
                        trans_probs[to_state] = (state_trans['state'] == to_state).mean()
                    matrix_dict[(program, term)] = trans_probs

    transition_matrices[from_state] = matrix_dict

    # Print sample rates
    sample_key = list(matrix_dict.keys())[0]
    sample_probs = matrix_dict[sample_key]
    print(f"    Sample (Program={sample_key[0]}, Term={sample_key[1]}):")
    print(f"      → Charged-off: {sample_probs.get('CHARGED_OFF', 0)*100:.2f}%")
    print(f"      → Paid-off: {sample_probs.get('PAID_OFF', 0)*100:.2f}%")
    print(f"      → Cure to Current: {sample_probs.get('CURRENT', 0)*100:.2f}%")
    print(f"      → Stay same state: {sample_probs.get(from_state, 0)*100:.2f}%")

# ============================================================================
# 5. COMPUTE OVERALL ROLL RATES BY STATE
# ============================================================================
print("\n5. Overall roll rates by delinquency state:")

for from_state in delinq_states + ['CURRENT']:
    state_trans = transitions[transitions['prev_state'] == from_state]
    if len(state_trans) > 0:
        to_co = (state_trans['state'] == 'CHARGED_OFF').mean()
        to_po = (state_trans['state'] == 'PAID_OFF').mean()
        to_curr = (state_trans['state'] == 'CURRENT').mean()
        stay_same = (state_trans['state'] == from_state).mean()

        print(f"\n  {from_state}:")
        print(f"    → Charge-off: {to_co*100:.2f}%")
        print(f"    → Paid-off: {to_po*100:.2f}%")
        print(f"    → Current (cure): {to_curr*100:.2f}%")
        print(f"    → Stay same: {stay_same*100:.2f}%")
        print(f"    Observations: {len(state_trans):,}")

# ============================================================================
# 6. SAVE MODELS
# ============================================================================
print("\n6. Saving hybrid transition models...")

import pickle

hybrid_models = {
    # Regression models (for CURRENT state only)
    'model_d1_29': model_d1_29,
    'scaler_d1_29': scaler_d1_29,
    'model_prepay': model_prepay,
    'scaler_prepay': scaler_prepay,

    # Feature columns for each model
    'feature_cols_d1_29': feature_cols,  # Features for D1-29 model (all categorical/dummies)
    'feature_cols_prepay': prepay_feature_cols,  # Features for Prepay model (all categorical/dummies)
    'numeric_features_d1_29': [],  # No numeric features - all categorical
    'numeric_features_prepay': [],  # No numeric features - all categorical

    # Empirical matrices (for all delinquency states)
    'transition_matrices': transition_matrices,
    'programs': programs,
    'term_buckets': term_buckets,

    # States
    'all_states': all_states,
    'delinq_states': delinq_states,

    # Model performance
    'auc_d1_29': auc_d1_29,
    'auc_prepay': auc_prepay,

    # Feature importance
    'feature_importance_d1_29': coefs_d1_29,
    'feature_importance_prepay': coefs_prepay
}

with open('hybrid_transition_models.pkl', 'wb') as f:
    pickle.dump(hybrid_models, f)

print("  ✓ Saved to hybrid_transition_models.pkl")

# Save feature importance to CSV for review
coefs_d1_29.to_csv('feature_importance_d1_29.csv', index=False)
coefs_prepay.to_csv('feature_importance_prepay.csv', index=False)
print("  ✓ Saved feature importance to CSV files")

# Combine D1-29 and Payoff predictions into single CSV
combined_predictions = pd.concat([d1_29_by_age, prepay_by_age], ignore_index=True)
combined_predictions = combined_predictions.sort_values(['model', 'sample', 'loan_age_months'])
combined_predictions.to_csv('current_state_predictions_by_age.csv', index=False)
print("  ✓ Saved combined prediction vs actual to current_state_predictions_by_age.csv")

# ============================================================================
# 7. VISUALIZE PREDICTION VS ACTUAL BY LOAN AGE
# ============================================================================
print("\n7. Creating prediction vs actual visualizations...")

import matplotlib.pyplot as plt

# ============================================================================
# Chart 1: Combined Current State Models (Side by Side)
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(20, 7))
fig.suptitle('Current State Transition Models: Prediction vs Actual by Loan Age',
             fontsize=16, fontweight='bold', y=0.98)

# Left: D1-29 Model (both train and test)
ax = axes[0]
train_data = d1_29_by_age[d1_29_by_age['sample'] == 'train'].copy()
test_data = d1_29_by_age[d1_29_by_age['sample'] == 'test'].copy()

ax.plot(train_data['loan_age_months'], train_data['actual_rate'] * 100,
        marker='o', linewidth=2, markersize=3, label='Train Actual', color='steelblue', alpha=0.7)
ax.plot(train_data['loan_age_months'], train_data['predicted_rate'] * 100,
        marker='', linewidth=2, label='Train Predicted', color='coral', linestyle='--', alpha=0.7)
ax.plot(test_data['loan_age_months'], test_data['actual_rate'] * 100,
        marker='s', linewidth=2, markersize=3, label='Test Actual', color='navy', alpha=0.7)
ax.plot(test_data['loan_age_months'], test_data['predicted_rate'] * 100,
        marker='', linewidth=2, label='Test Predicted', color='red', linestyle='--', alpha=0.7)

ax.set_xlabel('Loan Age (Months)', fontsize=12, fontweight='bold')
ax.set_ylabel('D1-29 Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('Current → D1-29 (Early Delinquency)', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='best', ncol=2)
ax.grid(alpha=0.3)
ax.tick_params(axis='x', labelsize=10)
ax.tick_params(axis='y', labelsize=10)

# Add sample size info
total_train = train_data['num_obs'].sum()
total_test = test_data['num_obs'].sum()
ax.text(0.02, 0.98, f"Train: {total_train:,} obs | Test: {total_test:,} obs",
        transform=ax.transAxes, fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

# Right: Prepay Model (both train and test)
ax = axes[1]
train_data = prepay_by_age[prepay_by_age['sample'] == 'train'].copy()
test_data = prepay_by_age[prepay_by_age['sample'] == 'test'].copy()

ax.plot(train_data['loan_age_months'], train_data['actual_rate'] * 100,
        marker='o', linewidth=2, markersize=3, label='Train Actual', color='forestgreen', alpha=0.7)
ax.plot(train_data['loan_age_months'], train_data['predicted_rate'] * 100,
        marker='', linewidth=2, label='Train Predicted', color='orange', linestyle='--', alpha=0.7)
ax.plot(test_data['loan_age_months'], test_data['actual_rate'] * 100,
        marker='s', linewidth=2, markersize=3, label='Test Actual', color='darkgreen', alpha=0.7)
ax.plot(test_data['loan_age_months'], test_data['predicted_rate'] * 100,
        marker='', linewidth=2, label='Test Predicted', color='darkorange', linestyle='--', alpha=0.7)

ax.set_xlabel('Loan Age (Months)', fontsize=12, fontweight='bold')
ax.set_ylabel('Payoff Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('Current → Payoff', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='best', ncol=2)
ax.grid(alpha=0.3)
ax.tick_params(axis='x', labelsize=10)
ax.tick_params(axis='y', labelsize=10)

# Add sample size info
total_train = train_data['num_obs'].sum()
total_test = test_data['num_obs'].sum()
ax.text(0.02, 0.98, f"Train: {total_train:,} obs | Test: {total_test:,} obs",
        transform=ax.transAxes, fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('current_state_models_combined.png', dpi=150, bbox_inches='tight')
print("  ✓ Saved combined visualization: current_state_models_combined.png")
plt.close()

# ============================================================================
# Chart 2: Program-Level Charts by Term (D1-29 and Prepay by Program × Term)
# ============================================================================
print("  Creating program-level visualizations by term...")

# Merge program info back to predictions
# D1-29 model - use the actual y values from the split
d1_29_y_train = y_d1_29.loc[train_idx]
d1_29_y_test = y_d1_29.loc[test_idx]

d1_29_results_with_program = pd.DataFrame({
    'loan_age_months': current_transitions.loc[train_idx, 'loan_age_months'].tolist() +
                       current_transitions.loc[test_idx, 'loan_age_months'].tolist(),
    'program': current_transitions.loc[train_idx, 'program'].tolist() +
               current_transitions.loc[test_idx, 'program'].tolist(),
    'loan_term': current_transitions.loc[train_idx, 'loan_term'].tolist() +
                 current_transitions.loc[test_idx, 'loan_term'].tolist(),
    'actual': d1_29_y_train.tolist() + d1_29_y_test.tolist(),
    'predicted_prob': y_train_pred_proba.tolist() + y_test_pred_proba.tolist(),
    'sample': ['train'] * len(train_idx) + ['test'] * len(test_idx),
    'model': 'Current_to_D1_29'
})

# Prepay model - use the actual y values from the split
prepay_y_train = y_prepay.loc[prepay_train_idx]
prepay_y_test = y_prepay.loc[prepay_test_idx]

prepay_results_with_program = pd.DataFrame({
    'loan_age_months': current_transitions.loc[prepay_train_idx, 'loan_age_months'].tolist() +
                       current_transitions.loc[prepay_test_idx, 'loan_age_months'].tolist(),
    'program': current_transitions.loc[prepay_train_idx, 'program'].tolist() +
               current_transitions.loc[prepay_test_idx, 'program'].tolist(),
    'loan_term': current_transitions.loc[prepay_train_idx, 'loan_term'].tolist() +
                 current_transitions.loc[prepay_test_idx, 'loan_term'].tolist(),
    'actual': prepay_y_train.tolist() + prepay_y_test.tolist(),
    'predicted_prob': y_train_pred_proba_prepay.tolist() + y_test_pred_proba_prepay.tolist(),
    'sample': ['train'] * len(prepay_train_idx) + ['test'] * len(prepay_test_idx),
    'model': 'Current_to_Prepay'
})

# Get unique programs and top terms
unique_programs = sorted(current_transitions['program'].unique())
num_programs = len(unique_programs)

# Get top 6 most common terms
common_terms = sorted(current_transitions['loan_term'].value_counts().head(6).index.tolist())
num_terms = len(common_terms)

# Create subplots: 2 columns (D1-29, Prepay) x (num_terms × num_programs) rows
# Each term gets a section with all programs
fig, axes = plt.subplots(num_terms * num_programs, 2, figsize=(20, 6 * num_terms * num_programs))
if num_terms * num_programs == 1:
    axes = axes.reshape(1, -1)

fig.suptitle('Current State Models by Program and Term: Prediction vs Actual',
             fontsize=18, fontweight='bold', y=0.998)

# Iterate through terms, then programs within each term
row_idx = 0
for term_idx, term in enumerate(common_terms):
    for prog_idx, program in enumerate(unique_programs):
        # D1-29 model (left column)
        ax = axes[row_idx, 0]

        # Filter data for this term and program
        term_prog_data = d1_29_results_with_program[
            (d1_29_results_with_program['loan_term'] == term) &
            (d1_29_results_with_program['program'] == program)
        ].copy()

        # Aggregate by loan age
        if len(term_prog_data) > 0:
            aggregated = term_prog_data.groupby(['loan_age_months', 'sample']).agg({
                'actual': ['mean', 'count'],
                'predicted_prob': 'mean'
            }).reset_index()
            aggregated.columns = ['loan_age_months', 'sample', 'actual_rate', 'num_obs', 'predicted_rate']
            # Filter to only show age >= 1 month
            aggregated = aggregated[aggregated['loan_age_months'] >= 1]

            train_data = aggregated[aggregated['sample'] == 'train']
            test_data = aggregated[aggregated['sample'] == 'test']

            # Plot
            if len(train_data) > 0:
                ax.plot(train_data['loan_age_months'], train_data['actual_rate'] * 100,
                        marker='o', linewidth=2, markersize=3, label='Train Actual', color='steelblue', alpha=0.7)
                ax.plot(train_data['loan_age_months'], train_data['predicted_rate'] * 100,
                        marker='', linewidth=2, label='Train Predicted', color='coral', linestyle='--', alpha=0.7)
            if len(test_data) > 0:
                ax.plot(test_data['loan_age_months'], test_data['actual_rate'] * 100,
                        marker='s', linewidth=2, markersize=3, label='Test Actual', color='navy', alpha=0.7)
                ax.plot(test_data['loan_age_months'], test_data['predicted_rate'] * 100,
                        marker='', linewidth=2, label='Test Predicted', color='red', linestyle='--', alpha=0.7)

            # Add sample size
            total_train = train_data['num_obs'].sum() if len(train_data) > 0 else 0
            total_test = test_data['num_obs'].sum() if len(test_data) > 0 else 0
            ax.text(0.02, 0.98, f"n={total_train:,}/{total_test:,}",
                    transform=ax.transAxes, fontsize=8, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        ax.set_xlabel('Loan Age (Months)', fontsize=10, fontweight='bold')
        ax.set_ylabel('D1-29 Rate (%)', fontsize=10, fontweight='bold')
        ax.set_title(f'Term={term}m, {program} - D1-29', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='best', ncol=2)
        ax.grid(alpha=0.3)
        ax.tick_params(axis='x', labelsize=9)
        ax.tick_params(axis='y', labelsize=9)

        # Prepay model (right column)
        ax = axes[row_idx, 1]

        # Filter data for this term and program
        term_prog_data = prepay_results_with_program[
            (prepay_results_with_program['loan_term'] == term) &
            (prepay_results_with_program['program'] == program)
        ].copy()

        # Aggregate by loan age
        if len(term_prog_data) > 0:
            aggregated = term_prog_data.groupby(['loan_age_months', 'sample']).agg({
                'actual': ['mean', 'count'],
                'predicted_prob': 'mean'
            }).reset_index()
            aggregated.columns = ['loan_age_months', 'sample', 'actual_rate', 'num_obs', 'predicted_rate']
            # Filter to only show age >= 1 month
            aggregated = aggregated[aggregated['loan_age_months'] >= 1]

            train_data = aggregated[aggregated['sample'] == 'train']
            test_data = aggregated[aggregated['sample'] == 'test']

            # Plot
            if len(train_data) > 0:
                ax.plot(train_data['loan_age_months'], train_data['actual_rate'] * 100,
                        marker='o', linewidth=2, markersize=3, label='Train Actual', color='forestgreen', alpha=0.7)
                ax.plot(train_data['loan_age_months'], train_data['predicted_rate'] * 100,
                        marker='', linewidth=2, label='Train Predicted', color='orange', linestyle='--', alpha=0.7)
            if len(test_data) > 0:
                ax.plot(test_data['loan_age_months'], test_data['actual_rate'] * 100,
                        marker='s', linewidth=2, markersize=3, label='Test Actual', color='darkgreen', alpha=0.7)
                ax.plot(test_data['loan_age_months'], test_data['predicted_rate'] * 100,
                        marker='', linewidth=2, label='Test Predicted', color='darkorange', linestyle='--', alpha=0.7)

            # Add sample size
            total_train = train_data['num_obs'].sum() if len(train_data) > 0 else 0
            total_test = test_data['num_obs'].sum() if len(test_data) > 0 else 0
            ax.text(0.02, 0.98, f"n={total_train:,}/{total_test:,}",
                    transform=ax.transAxes, fontsize=8, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        ax.set_xlabel('Loan Age (Months)', fontsize=10, fontweight='bold')
        ax.set_ylabel('Payoff Rate (%)', fontsize=10, fontweight='bold')
        ax.set_title(f'Term={term}m, {program} - Payoff', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='best', ncol=2)
        ax.grid(alpha=0.3)
        ax.tick_params(axis='x', labelsize=9)
        ax.tick_params(axis='y', labelsize=9)

        row_idx += 1

plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('current_state_models_by_program.png', dpi=150, bbox_inches='tight')
print("  ✓ Saved program-level visualization: current_state_models_by_program.png")
plt.close()

# ============================================================================
# Chart 3: By Vintage (Age Buckets)
# ============================================================================
print("  Creating vintage-level visualizations...")

# Add age buckets to results
d1_29_results_with_vintage = d1_29_results_with_program.copy()
d1_29_results_with_vintage['vintage'] = d1_29_results_with_vintage['loan_age_months'].apply(create_age_buckets)

prepay_results_with_vintage = prepay_results_with_program.copy()
prepay_results_with_vintage['vintage'] = prepay_results_with_vintage['loan_age_months'].apply(create_age_buckets)

# Aggregate by vintage and sample
d1_29_by_vintage = d1_29_results_with_vintage.groupby(['vintage', 'sample']).agg({
    'actual': ['mean', 'count'],
    'predicted_prob': 'mean'
}).reset_index()
d1_29_by_vintage.columns = ['vintage', 'sample', 'actual_rate', 'num_obs', 'predicted_rate']

prepay_by_vintage = prepay_results_with_vintage.groupby(['vintage', 'sample']).agg({
    'actual': ['mean', 'count'],
    'predicted_prob': 'mean'
}).reset_index()
prepay_by_vintage.columns = ['vintage', 'sample', 'actual_rate', 'num_obs', 'predicted_rate']

# Define vintage order
vintage_order = ['0-1m', '2-3m', '4-6m', '7-12m', '13-18m', '19-24m', '24m+']
d1_29_by_vintage['vintage'] = pd.Categorical(d1_29_by_vintage['vintage'], categories=vintage_order, ordered=True)
prepay_by_vintage['vintage'] = pd.Categorical(prepay_by_vintage['vintage'], categories=vintage_order, ordered=True)
d1_29_by_vintage = d1_29_by_vintage.sort_values('vintage')
prepay_by_vintage = prepay_by_vintage.sort_values('vintage')

# Create chart
fig, axes = plt.subplots(1, 2, figsize=(20, 7))
fig.suptitle('Current State Models by Vintage (Age Bucket): Prediction vs Actual',
             fontsize=16, fontweight='bold', y=0.98)

# D1-29 by vintage
ax = axes[0]
train_data = d1_29_by_vintage[d1_29_by_vintage['sample'] == 'train']
test_data = d1_29_by_vintage[d1_29_by_vintage['sample'] == 'test']

x_pos = np.arange(len(vintage_order))
width = 0.35

ax.bar(x_pos - width/2, train_data['actual_rate'] * 100, width,
       label='Train Actual', color='steelblue', alpha=0.7)
ax.bar(x_pos + width/2, test_data['actual_rate'] * 100, width,
       label='Test Actual', color='navy', alpha=0.7)
ax.plot(x_pos, train_data['predicted_rate'] * 100, marker='o', linewidth=2,
        markersize=8, label='Train Predicted', color='coral', linestyle='--')
ax.plot(x_pos, test_data['predicted_rate'] * 100, marker='s', linewidth=2,
        markersize=8, label='Test Predicted', color='red', linestyle='--')

ax.set_xlabel('Vintage (Age Bucket)', fontsize=12, fontweight='bold')
ax.set_ylabel('D1-29 Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('Current → D1-29 by Vintage', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(vintage_order, fontsize=10)
ax.legend(fontsize=10, loc='best')
ax.grid(alpha=0.3, axis='y')

# Add sample counts
for i, vintage in enumerate(vintage_order):
    train_count = train_data[train_data['vintage'] == vintage]['num_obs'].values
    test_count = test_data[test_data['vintage'] == vintage]['num_obs'].values
    if len(train_count) > 0 and len(test_count) > 0:
        ax.text(i, -0.5, f"n={train_count[0]:,}/{test_count[0]:,}",
                ha='center', fontsize=8, rotation=0)

# Prepay by vintage
ax = axes[1]
train_data = prepay_by_vintage[prepay_by_vintage['sample'] == 'train']
test_data = prepay_by_vintage[prepay_by_vintage['sample'] == 'test']

ax.bar(x_pos - width/2, train_data['actual_rate'] * 100, width,
       label='Train Actual', color='forestgreen', alpha=0.7)
ax.bar(x_pos + width/2, test_data['actual_rate'] * 100, width,
       label='Test Actual', color='darkgreen', alpha=0.7)
ax.plot(x_pos, train_data['predicted_rate'] * 100, marker='o', linewidth=2,
        markersize=8, label='Train Predicted', color='orange', linestyle='--')
ax.plot(x_pos, test_data['predicted_rate'] * 100, marker='s', linewidth=2,
        markersize=8, label='Test Predicted', color='darkorange', linestyle='--')

ax.set_xlabel('Vintage (Age Bucket)', fontsize=12, fontweight='bold')
ax.set_ylabel('Payoff Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('Current → Payoff by Vintage', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(vintage_order, fontsize=10)
ax.legend(fontsize=10, loc='best')
ax.grid(alpha=0.3, axis='y')

# Add sample counts
for i, vintage in enumerate(vintage_order):
    train_count = train_data[train_data['vintage'] == vintage]['num_obs'].values
    test_count = test_data[test_data['vintage'] == vintage]['num_obs'].values
    if len(train_count) > 0 and len(test_count) > 0:
        ax.text(i, -0.5, f"n={train_count[0]:,}/{test_count[0]:,}",
                ha='center', fontsize=8, rotation=0)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('current_state_models_by_age_bucket.png', dpi=150, bbox_inches='tight')
print("  ✓ Saved age bucket visualization: current_state_models_by_age_bucket.png")
plt.close()

# ============================================================================
# Chart 4: By Vintage Quarter (Disbursement Quarter, excluding 2019) - BY PROGRAM
# ============================================================================
print("  Creating vintage quarter visualizations by program...")

# Extract disbursement quarter from transitions
d1_29_results_with_vintage_qtr = d1_29_results_with_program.copy()
d1_29_results_with_vintage_qtr['disbursement_d'] = current_transitions.loc[train_idx, 'disbursement_d'].tolist() + \
                                                     current_transitions.loc[test_idx, 'disbursement_d'].tolist()
d1_29_results_with_vintage_qtr['disbursement_dt'] = pd.to_datetime(d1_29_results_with_vintage_qtr['disbursement_d'])
d1_29_results_with_vintage_qtr['disbursement_year'] = d1_29_results_with_vintage_qtr['disbursement_dt'].dt.year
d1_29_results_with_vintage_qtr['disbursement_quarter'] = d1_29_results_with_vintage_qtr['disbursement_dt'].dt.to_period('Q').astype(str)

prepay_results_with_vintage_qtr = prepay_results_with_program.copy()
prepay_results_with_vintage_qtr['disbursement_d'] = current_transitions.loc[prepay_train_idx, 'disbursement_d'].tolist() + \
                                                      current_transitions.loc[prepay_test_idx, 'disbursement_d'].tolist()
prepay_results_with_vintage_qtr['disbursement_dt'] = pd.to_datetime(prepay_results_with_vintage_qtr['disbursement_d'])
prepay_results_with_vintage_qtr['disbursement_year'] = prepay_results_with_vintage_qtr['disbursement_dt'].dt.year
prepay_results_with_vintage_qtr['disbursement_quarter'] = prepay_results_with_vintage_qtr['disbursement_dt'].dt.to_period('Q').astype(str)

# Exclude 2019
d1_29_results_with_vintage_qtr = d1_29_results_with_vintage_qtr[d1_29_results_with_vintage_qtr['disbursement_year'] >= 2020]
prepay_results_with_vintage_qtr = prepay_results_with_vintage_qtr[prepay_results_with_vintage_qtr['disbursement_year'] >= 2020]

# Aggregate by disbursement quarter, program, and sample
d1_29_by_vintage_qtr = d1_29_results_with_vintage_qtr.groupby(['disbursement_quarter', 'program', 'sample']).agg({
    'actual': ['mean', 'count'],
    'predicted_prob': 'mean'
}).reset_index()
d1_29_by_vintage_qtr.columns = ['disbursement_quarter', 'program', 'sample', 'actual_rate', 'num_obs', 'predicted_rate']
d1_29_by_vintage_qtr = d1_29_by_vintage_qtr.sort_values(['program', 'disbursement_quarter'])

prepay_by_vintage_qtr = prepay_results_with_vintage_qtr.groupby(['disbursement_quarter', 'program', 'sample']).agg({
    'actual': ['mean', 'count'],
    'predicted_prob': 'mean'
}).reset_index()
prepay_by_vintage_qtr.columns = ['disbursement_quarter', 'program', 'sample', 'actual_rate', 'num_obs', 'predicted_rate']
prepay_by_vintage_qtr = prepay_by_vintage_qtr.sort_values(['program', 'disbursement_quarter'])

# Get unique programs and quarters
programs = sorted(d1_29_results_with_vintage_qtr['program'].unique())
unique_quarters = sorted(d1_29_by_vintage_qtr['disbursement_quarter'].unique())

# Create chart with 3 rows (one per program) and 2 columns (D1-29, Prepay)
fig, axes = plt.subplots(3, 2, figsize=(20, 18))
fig.suptitle('Current State Models by Vintage Quarter (Disbursement Quarter, 2020+) by Program: Prediction vs Actual',
             fontsize=16, fontweight='bold', y=0.995)

for row_idx, program in enumerate(programs):
    # D1-29 by vintage quarter for this program
    ax = axes[row_idx, 0]

    prog_data = d1_29_by_vintage_qtr[d1_29_by_vintage_qtr['program'] == program]
    train_data = prog_data[prog_data['sample'] == 'train']
    test_data = prog_data[prog_data['sample'] == 'test']

    # Get quarters for this program
    prog_quarters = sorted(prog_data['disbursement_quarter'].unique())
    x_pos = np.arange(len(prog_quarters))
    width = 0.35

    if len(train_data) > 0 and len(test_data) > 0:
        ax.bar(x_pos - width/2, train_data['actual_rate'] * 100, width,
               label='Train Actual', color='steelblue', alpha=0.7)
        ax.bar(x_pos + width/2, test_data['actual_rate'] * 100, width,
               label='Test Actual', color='navy', alpha=0.7)
        ax.plot(x_pos, train_data['predicted_rate'] * 100, marker='o', linewidth=2,
                markersize=8, label='Train Predicted', color='coral', linestyle='--')
        ax.plot(x_pos, test_data['predicted_rate'] * 100, marker='s', linewidth=2,
                markersize=8, label='Test Predicted', color='red', linestyle='--')

        # Add sample counts
        for i, qtr in enumerate(prog_quarters):
            train_count = train_data[train_data['disbursement_quarter'] == qtr]['num_obs'].values
            test_count = test_data[test_data['disbursement_quarter'] == qtr]['num_obs'].values
            if len(train_count) > 0 and len(test_count) > 0:
                ax.text(i, -0.5, f"n={train_count[0]:,}/{test_count[0]:,}",
                        ha='center', fontsize=7, rotation=45)

    ax.set_xlabel('Vintage Quarter (Disbursement Quarter)', fontsize=11, fontweight='bold')
    ax.set_ylabel('D1-29 Rate (%)', fontsize=11, fontweight='bold')
    ax.set_title(f'{program} - Current → D1-29 by Vintage Quarter', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(prog_quarters, fontsize=8, rotation=45, ha='right')
    ax.legend(fontsize=9, loc='best')
    ax.grid(alpha=0.3, axis='y')

    # Prepay by vintage quarter for this program
    ax = axes[row_idx, 1]

    prog_data = prepay_by_vintage_qtr[prepay_by_vintage_qtr['program'] == program]
    train_data = prog_data[prog_data['sample'] == 'train']
    test_data = prog_data[prog_data['sample'] == 'test']

    # Get quarters for this program
    prog_quarters = sorted(prog_data['disbursement_quarter'].unique())
    x_pos = np.arange(len(prog_quarters))

    if len(train_data) > 0 and len(test_data) > 0:
        ax.bar(x_pos - width/2, train_data['actual_rate'] * 100, width,
               label='Train Actual', color='forestgreen', alpha=0.7)
        ax.bar(x_pos + width/2, test_data['actual_rate'] * 100, width,
               label='Test Actual', color='darkgreen', alpha=0.7)
        ax.plot(x_pos, train_data['predicted_rate'] * 100, marker='o', linewidth=2,
                markersize=8, label='Train Predicted', color='orange', linestyle='--')
        ax.plot(x_pos, test_data['predicted_rate'] * 100, marker='s', linewidth=2,
                markersize=8, label='Test Predicted', color='darkorange', linestyle='--')

        # Add sample counts
        for i, qtr in enumerate(prog_quarters):
            train_count = train_data[train_data['disbursement_quarter'] == qtr]['num_obs'].values
            test_count = test_data[test_data['disbursement_quarter'] == qtr]['num_obs'].values
            if len(train_count) > 0 and len(test_count) > 0:
                ax.text(i, -0.5, f"n={train_count[0]:,}/{test_count[0]:,}",
                        ha='center', fontsize=7, rotation=45)

    ax.set_xlabel('Vintage Quarter (Disbursement Quarter)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Payoff Rate (%)', fontsize=11, fontweight='bold')
    ax.set_title(f'{program} - Current → Payoff by Vintage Quarter', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(prog_quarters, fontsize=8, rotation=45, ha='right')
    ax.legend(fontsize=9, loc='best')
    ax.grid(alpha=0.3, axis='y')

plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('current_state_models_by_vintage.png', dpi=150, bbox_inches='tight')
print("  ✓ Saved vintage quarter visualization by program: current_state_models_by_vintage.png")
plt.close()

# ============================================================================
# Chart 5: By Term
# ============================================================================
print("  Creating term-level visualizations...")

# Add term to results
d1_29_results_with_term = d1_29_results_with_program.copy()
d1_29_results_with_term['term'] = current_transitions.loc[train_idx, 'loan_term'].tolist() + \
                                    current_transitions.loc[test_idx, 'loan_term'].tolist()

prepay_results_with_term = prepay_results_with_program.copy()
prepay_results_with_term['term'] = current_transitions.loc[prepay_train_idx, 'loan_term'].tolist() + \
                                     current_transitions.loc[prepay_test_idx, 'loan_term'].tolist()

# Get most common terms
common_terms = sorted(current_transitions['loan_term'].value_counts().head(6).index.tolist())

# Filter to common terms
d1_29_by_term_filtered = d1_29_results_with_term[d1_29_results_with_term['term'].isin(common_terms)]
prepay_by_term_filtered = prepay_results_with_term[prepay_results_with_term['term'].isin(common_terms)]

# Aggregate by term and sample
d1_29_by_term = d1_29_by_term_filtered.groupby(['term', 'sample']).agg({
    'actual': ['mean', 'count'],
    'predicted_prob': 'mean'
}).reset_index()
d1_29_by_term.columns = ['term', 'sample', 'actual_rate', 'num_obs', 'predicted_rate']
d1_29_by_term = d1_29_by_term.sort_values('term')

prepay_by_term = prepay_by_term_filtered.groupby(['term', 'sample']).agg({
    'actual': ['mean', 'count'],
    'predicted_prob': 'mean'
}).reset_index()
prepay_by_term.columns = ['term', 'sample', 'actual_rate', 'num_obs', 'predicted_rate']
prepay_by_term = prepay_by_term.sort_values('term')

# Create chart
fig, axes = plt.subplots(1, 2, figsize=(20, 7))
fig.suptitle('Current State Models by Loan Term: Prediction vs Actual',
             fontsize=16, fontweight='bold', y=0.98)

# D1-29 by term
ax = axes[0]
train_data = d1_29_by_term[d1_29_by_term['sample'] == 'train']
test_data = d1_29_by_term[d1_29_by_term['sample'] == 'test']

x_pos = np.arange(len(common_terms))
width = 0.35

ax.bar(x_pos - width/2, train_data['actual_rate'] * 100, width,
       label='Train Actual', color='steelblue', alpha=0.7)
ax.bar(x_pos + width/2, test_data['actual_rate'] * 100, width,
       label='Test Actual', color='navy', alpha=0.7)
ax.plot(x_pos, train_data['predicted_rate'] * 100, marker='o', linewidth=2,
        markersize=8, label='Train Predicted', color='coral', linestyle='--')
ax.plot(x_pos, test_data['predicted_rate'] * 100, marker='s', linewidth=2,
        markersize=8, label='Test Predicted', color='red', linestyle='--')

ax.set_xlabel('Loan Term (Months)', fontsize=12, fontweight='bold')
ax.set_ylabel('D1-29 Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('Current → D1-29 by Term', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels([f"{t}m" for t in common_terms], fontsize=10)
ax.legend(fontsize=10, loc='best')
ax.grid(alpha=0.3, axis='y')

# Add sample counts
for i, term in enumerate(common_terms):
    train_count = train_data[train_data['term'] == term]['num_obs'].values
    test_count = test_data[test_data['term'] == term]['num_obs'].values
    if len(train_count) > 0 and len(test_count) > 0:
        ax.text(i, -0.5, f"n={train_count[0]:,}/{test_count[0]:,}",
                ha='center', fontsize=8, rotation=0)

# Prepay by term
ax = axes[1]
train_data = prepay_by_term[prepay_by_term['sample'] == 'train']
test_data = prepay_by_term[prepay_by_term['sample'] == 'test']

ax.bar(x_pos - width/2, train_data['actual_rate'] * 100, width,
       label='Train Actual', color='forestgreen', alpha=0.7)
ax.bar(x_pos + width/2, test_data['actual_rate'] * 100, width,
       label='Test Actual', color='darkgreen', alpha=0.7)
ax.plot(x_pos, train_data['predicted_rate'] * 100, marker='o', linewidth=2,
        markersize=8, label='Train Predicted', color='orange', linestyle='--')
ax.plot(x_pos, test_data['predicted_rate'] * 100, marker='s', linewidth=2,
        markersize=8, label='Test Predicted', color='darkorange', linestyle='--')

ax.set_xlabel('Loan Term (Months)', fontsize=12, fontweight='bold')
ax.set_ylabel('Payoff Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('Current → Payoff by Term', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels([f"{t}m" for t in common_terms], fontsize=10)
ax.legend(fontsize=10, loc='best')
ax.grid(alpha=0.3, axis='y')

# Add sample counts
for i, term in enumerate(common_terms):
    train_count = train_data[train_data['term'] == term]['num_obs'].values
    test_count = test_data[test_data['term'] == term]['num_obs'].values
    if len(train_count) > 0 and len(test_count) > 0:
        ax.text(i, -0.5, f"n={train_count[0]:,}/{test_count[0]:,}",
                ha='center', fontsize=8, rotation=0)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('current_state_models_by_term.png', dpi=150, bbox_inches='tight')
print("  ✓ Saved term-level visualization: current_state_models_by_term.png")
plt.close()

print("\n" + "="*80)
print("HYBRID TRANSITION MODEL BUILD COMPLETE")
print("="*80)
print("\nModel Summary:")
print(f"  Dataset: loan_performance_enhanced.csv")
print(f"  Unique loans: {df['display_id'].nunique():,}")
print(f"  Total transitions analyzed: {len(transitions):,}")
print(f"\n  Regression Models (for CURRENT state):")
print(f"    • Current → D1-29 AUC: {auc_d1_29:.4f}")
print(f"      - Features ({len(feature_cols)}): FICO, amount, term, age, UPB, payments, delinq history, program")
print(f"    • Current → Payoff AUC: {auc_prepay:.4f}")
print(f"      - Features ({len(prepay_feature_cols)}): program, loan_term, loan_age_months ONLY")
print(f"\n  Empirical Matrices:")
print(f"    • Delinquency states covered: {len(transition_matrices)}")
print(f"    • Matrix dimensions: {len(programs)} Programs x {len(term_buckets)} Term buckets")
print(f"\n  Feature Strategy:")
print(f"    • D1-29 model: Full feature set with delinquency history")
print(f"    • Payoff model: Simplified - program, term, age only")
print(f"\n  Output Files Generated:")
print(f"    Models:")
print(f"      • hybrid_transition_models.pkl - Model objects and matrices")
print(f"    Feature Importance:")
print(f"      • feature_importance_d1_29.csv - D1-29 feature coefficients")
print(f"      • feature_importance_prepay.csv - Payoff feature coefficients")
print(f"    Predictions:")
print(f"      • current_state_predictions_by_age.csv - Combined D1-29 & Payoff predictions by loan age")
print(f"    Visualizations:")
print(f"      • current_state_models_combined.png - Overall side-by-side comparison (1x2)")
print(f"      • current_state_models_by_program.png - Program-level breakdown")
print(f"      • current_state_models_by_age_bucket.png - Age bucket breakdown")
print(f"      • current_state_models_by_vintage.png - Vintage quarter (disbursement quarter, 2020+) breakdown")
print(f"      • current_state_models_by_term.png - Loan term breakdown")
print("="*80)
