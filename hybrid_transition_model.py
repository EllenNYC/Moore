#!/usr/bin/env python3
"""
Hybrid Transition Model for Consumer Credit Portfolio
- Uses loan_performance_enhanced.csv with pre-computed features
- Regression for Current → D30 and Current → Prepay
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
    '1-30 DPD': 'D1_29',
    '31-60 DPD': 'D30_59',
    '61-90 DPD': 'D60_89',
    '91-120 DPD': 'D90_119',
    '120+ DPD': 'D120_PLUS'
}

# For terminal states, check loan_status
def assign_state(row):
    """Assign delinquency state"""
    if row['loan_status'] in ['CHARGED_OFF']:
        return 'CHARGED_OFF'
    elif row['loan_status'] in ['PAID_OFF']:
        return 'PAID_OFF'
    else:
        # Use delinquency_bucket for active loans
        return state_mapping.get(row['delinquency_bucket'], 'CURRENT')

df['state'] = df.apply(assign_state, axis=1)

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
# 3. BUILD REGRESSION MODELS (CURRENT → D30 AND CURRENT → PREPAY)
# ============================================================================
print("\n3. Building regression models for Current state transitions...")

# Filter to transitions from CURRENT state only
current_transitions = transitions[transitions['prev_state'] == 'CURRENT'].copy()

print(f"  Current-state transitions: {len(current_transitions):,}")
print(f"  Next state distribution:")
print(current_transitions['state'].value_counts())

# Create outcome variables
current_transitions['to_d30'] = current_transitions['state'].isin(
    ['D1_29', 'D30_59', 'D60_89', 'D90_119', 'D120_PLUS']
).astype(int)
current_transitions['to_prepay'] = (current_transitions['state'] == 'PAID_OFF').astype(int)
current_transitions['to_chargeoff'] = (current_transitions['state'] == 'CHARGED_OFF').astype(int)

print(f"\n  Outcome rates:")
print(f"    → Delinquency (D30+): {current_transitions['to_d30'].mean()*100:.2f}%")
print(f"    → Prepay: {current_transitions['to_prepay'].mean()*100:.2f}%")
print(f"    → Direct charge-off: {current_transitions['to_chargeoff'].mean()*100:.2f}%")

# Enhanced features from the dataset
numeric_features = [
    'fico_score',
    'approved_amount',
    'loan_term',
    'loan_age_months',
    'upb',
    'ever_D30',  # Has loan ever been 30+ DPD before?
    'ever_D60',  # Has loan ever been 60+ DPD before?
    'ever_D90'   # Has loan ever been 90+ DPD before?
]

# Add program dummies
program_dummies = pd.get_dummies(current_transitions['program'], prefix='program', drop_first=True)

# Combine features
X = pd.concat([current_transitions[numeric_features], program_dummies], axis=1)

# ============================================================================
# Model A: Current → D30+ (First Delinquency)
# ============================================================================
print("\n  Model A: Current → D30+ (First Delinquency)...")

y_d30 = current_transitions['to_d30']
valid_idx = X.notna().all(axis=1) & y_d30.notna()
X_clean = X[valid_idx]
y_clean = y_d30[valid_idx]

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

scaler_d30 = StandardScaler()
X_train_scaled = scaler_d30.fit_transform(X_train)
X_test_scaled = scaler_d30.transform(X_test)

model_d30 = LogisticRegression(random_state=42, max_iter=1000)
model_d30.fit(X_train_scaled, y_train)

# Evaluate
y_train_pred_proba = model_d30.predict_proba(X_train_scaled)[:, 1]
y_test_pred_proba = model_d30.predict_proba(X_test_scaled)[:, 1]
y_pred = model_d30.predict(X_test_scaled)
auc_d30 = roc_auc_score(y_test, y_test_pred_proba)

print(f"    AUC-ROC: {auc_d30:.4f}")
print(f"    Test accuracy: {(y_pred == y_test).mean()*100:.2f}%")

# Feature importance
feature_cols = list(X_clean.columns)
coefs_d30 = pd.DataFrame({
    'Feature': feature_cols,
    'Coefficient': model_d30.coef_[0]
}).sort_values('Coefficient', key=abs, ascending=False)

print(f"\n    Top 5 features (by absolute coefficient):")
for idx, row in coefs_d30.head(5).iterrows():
    print(f"      {row['Feature']:20s}: {row['Coefficient']:+.4f}")

# Create prediction vs actual by loan age (D30 model)
d30_train_results = pd.DataFrame({
    'loan_age_months': current_transitions.loc[train_idx, 'loan_age_months'],
    'actual': y_train,
    'predicted_prob': y_train_pred_proba,
    'sample': 'train',
    'model': 'Current_to_D30'
})

d30_test_results = pd.DataFrame({
    'loan_age_months': current_transitions.loc[test_idx, 'loan_age_months'],
    'actual': y_test,
    'predicted_prob': y_test_pred_proba,
    'sample': 'test',
    'model': 'Current_to_D30'
})

d30_results_combined = pd.concat([d30_train_results, d30_test_results], ignore_index=True)

# Aggregate by loan_age_months and sample
d30_by_age = d30_results_combined.groupby(['loan_age_months', 'sample']).agg({
    'actual': ['mean', 'count'],
    'predicted_prob': 'mean'
}).reset_index()

d30_by_age.columns = ['loan_age_months', 'sample', 'actual_rate', 'num_obs', 'predicted_rate']
d30_by_age['model'] = 'Current_to_D30'
d30_by_age = d30_by_age.sort_values(['loan_age_months', 'sample'])

print(f"\n    Prediction vs Actual by Loan Age (D30)")

# ============================================================================
# Model B: Current → Prepaid
# ============================================================================
print("\n  Model B: Current → Prepaid...")
print("    Using only: program, loan_term, loan_age_months")

# Simplified features for prepayment: only program, term, and loan age
prepay_numeric_features = ['loan_term', 'loan_age_months']
prepay_program_dummies = pd.get_dummies(current_transitions['program'], prefix='program', drop_first=True)
X_prepay = pd.concat([current_transitions[prepay_numeric_features], prepay_program_dummies], axis=1)

y_prepay = current_transitions['to_prepay']
valid_idx = X_prepay.notna().all(axis=1) & y_prepay.notna()
X_prepay_clean = X_prepay[valid_idx]
y_clean = y_prepay[valid_idx]

print(f"    Valid observations: {len(X_prepay_clean):,}")
print(f"    Positive class rate: {y_clean.mean()*100:.2f}%")
print(f"    Features: {len(X_prepay_clean.columns)} ({len(prepay_numeric_features)} numeric + {len(prepay_program_dummies.columns)} program dummies)")

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
prepay_by_age = prepay_by_age.sort_values(['loan_age_months', 'sample'])

print(f"\n    Prediction vs Actual by Loan Age (Prepay)")

# ============================================================================
# 4. BUILD EMPIRICAL MATRICES (PROGRAM x TERM) FOR OTHER TRANSITIONS
# ============================================================================
print("\n4. Building empirical transition matrices (Program x Loan Term)...")

# Get unique programs
programs = sorted(transitions['program'].unique())
print(f"  Programs: {programs}")

# Define term buckets
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

transitions['term_bucket'] = transitions['loan_term'].apply(categorize_term)
term_buckets = ['0-3m', '4-6m', '7-12m', '13-18m', '19-24m', '24m+']

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
    'model_d30': model_d30,
    'scaler_d30': scaler_d30,
    'model_prepay': model_prepay,
    'scaler_prepay': scaler_prepay,

    # Feature columns for each model
    'feature_cols_d30': feature_cols,  # Features for D30+ model
    'feature_cols_prepay': prepay_feature_cols,  # Features for Prepay model (program, term, age only)
    'numeric_features_d30': numeric_features,
    'numeric_features_prepay': prepay_numeric_features,

    # Empirical matrices (for all delinquency states)
    'transition_matrices': transition_matrices,
    'programs': programs,
    'term_buckets': term_buckets,

    # States
    'all_states': all_states,
    'delinq_states': delinq_states,

    # Model performance
    'auc_d30': auc_d30,
    'auc_prepay': auc_prepay,

    # Feature importance
    'feature_importance_d30': coefs_d30,
    'feature_importance_prepay': coefs_prepay
}

with open('hybrid_transition_models.pkl', 'wb') as f:
    pickle.dump(hybrid_models, f)

print("  ✓ Saved to hybrid_transition_models.pkl")

# Save feature importance to CSV for review
coefs_d30.to_csv('feature_importance_d30.csv', index=False)
coefs_prepay.to_csv('feature_importance_prepay.csv', index=False)
print("  ✓ Saved feature importance to CSV files")

# Combine D30 and Prepay predictions into single CSV
combined_predictions = pd.concat([d30_by_age, prepay_by_age], ignore_index=True)
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

# Left: D30+ Model (both train and test)
ax = axes[0]
train_data = d30_by_age[d30_by_age['sample'] == 'train'].copy()
test_data = d30_by_age[d30_by_age['sample'] == 'test'].copy()

ax.plot(train_data['loan_age_months'], train_data['actual_rate'] * 100,
        marker='o', linewidth=2, markersize=3, label='Train Actual', color='steelblue', alpha=0.7)
ax.plot(train_data['loan_age_months'], train_data['predicted_rate'] * 100,
        marker='', linewidth=2, label='Train Predicted', color='coral', linestyle='--', alpha=0.7)
ax.plot(test_data['loan_age_months'], test_data['actual_rate'] * 100,
        marker='s', linewidth=2, markersize=3, label='Test Actual', color='navy', alpha=0.7)
ax.plot(test_data['loan_age_months'], test_data['predicted_rate'] * 100,
        marker='', linewidth=2, label='Test Predicted', color='red', linestyle='--', alpha=0.7)

ax.set_xlabel('Loan Age (Months)', fontsize=12, fontweight='bold')
ax.set_ylabel('D30+ Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('Current → D30+ (Delinquency)', fontsize=14, fontweight='bold')
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
ax.set_ylabel('Prepay Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('Current → Prepay', fontsize=14, fontweight='bold')
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
# Chart 2: Program-Level Charts (D30+ and Prepay by Program)
# ============================================================================
print("  Creating program-level visualizations...")

# Merge program info back to predictions
# D30+ model - use the actual y values from the split
d30_y_train = y_d30.loc[train_idx]
d30_y_test = y_d30.loc[test_idx]

d30_results_with_program = pd.DataFrame({
    'loan_age_months': current_transitions.loc[train_idx, 'loan_age_months'].tolist() +
                       current_transitions.loc[test_idx, 'loan_age_months'].tolist(),
    'program': current_transitions.loc[train_idx, 'program'].tolist() +
               current_transitions.loc[test_idx, 'program'].tolist(),
    'actual': d30_y_train.tolist() + d30_y_test.tolist(),
    'predicted_prob': y_train_pred_proba.tolist() + y_test_pred_proba.tolist(),
    'sample': ['train'] * len(train_idx) + ['test'] * len(test_idx),
    'model': 'Current_to_D30'
})

# Prepay model - use the actual y values from the split
prepay_y_train = y_prepay.loc[prepay_train_idx]
prepay_y_test = y_prepay.loc[prepay_test_idx]

prepay_results_with_program = pd.DataFrame({
    'loan_age_months': current_transitions.loc[prepay_train_idx, 'loan_age_months'].tolist() +
                       current_transitions.loc[prepay_test_idx, 'loan_age_months'].tolist(),
    'program': current_transitions.loc[prepay_train_idx, 'program'].tolist() +
               current_transitions.loc[prepay_test_idx, 'program'].tolist(),
    'actual': prepay_y_train.tolist() + prepay_y_test.tolist(),
    'predicted_prob': y_train_pred_proba_prepay.tolist() + y_test_pred_proba_prepay.tolist(),
    'sample': ['train'] * len(prepay_train_idx) + ['test'] * len(prepay_test_idx),
    'model': 'Current_to_Prepay'
})

# Aggregate by loan_age_months, sample, and program
d30_by_program = d30_results_with_program.groupby(['loan_age_months', 'sample', 'program']).agg({
    'actual': ['mean', 'count'],
    'predicted_prob': 'mean'
}).reset_index()
d30_by_program.columns = ['loan_age_months', 'sample', 'program', 'actual_rate', 'num_obs', 'predicted_rate']

prepay_by_program = prepay_results_with_program.groupby(['loan_age_months', 'sample', 'program']).agg({
    'actual': ['mean', 'count'],
    'predicted_prob': 'mean'
}).reset_index()
prepay_by_program.columns = ['loan_age_months', 'sample', 'program', 'actual_rate', 'num_obs', 'predicted_rate']

# Get unique programs
unique_programs = sorted(current_transitions['program'].unique())
num_programs = len(unique_programs)

# Create subplots: 2 columns (D30+, Prepay) x N rows (one per program)
fig, axes = plt.subplots(num_programs, 2, figsize=(20, 6 * num_programs))
if num_programs == 1:
    axes = axes.reshape(1, -1)

fig.suptitle('Current State Models by Program: Prediction vs Actual',
             fontsize=18, fontweight='bold', y=0.995)

# D30+ models by program (left column)
for i, program in enumerate(unique_programs):
    ax = axes[i, 0]

    # Filter data for this program
    prog_data = d30_by_program[d30_by_program['program'] == program].copy()
    train_data = prog_data[prog_data['sample'] == 'train']
    test_data = prog_data[prog_data['sample'] == 'test']

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

    ax.set_xlabel('Loan Age (Months)', fontsize=11, fontweight='bold')
    ax.set_ylabel('D30+ Rate (%)', fontsize=11, fontweight='bold')
    ax.set_title(f'{program} - D30+', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='best', ncol=2)
    ax.grid(alpha=0.3)
    ax.tick_params(axis='x', labelsize=9)

    # Add sample size
    total_train = train_data['num_obs'].sum() if len(train_data) > 0 else 0
    total_test = test_data['num_obs'].sum() if len(test_data) > 0 else 0
    ax.text(0.02, 0.98, f"Train: {total_train:,} | Test: {total_test:,}",
            transform=ax.transAxes, fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

# Prepay models by program (right column)
for i, program in enumerate(unique_programs):
    ax = axes[i, 1]

    # Filter data for this program
    prog_data = prepay_by_program[prepay_by_program['program'] == program].copy()
    train_data = prog_data[prog_data['sample'] == 'train']
    test_data = prog_data[prog_data['sample'] == 'test']

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

    ax.set_xlabel('Loan Age (Months)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Prepay Rate (%)', fontsize=11, fontweight='bold')
    ax.set_title(f'{program} - Prepay', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='best', ncol=2)
    ax.grid(alpha=0.3)
    ax.tick_params(axis='x', labelsize=9)

    # Add sample size
    total_train = train_data['num_obs'].sum() if len(train_data) > 0 else 0
    total_test = test_data['num_obs'].sum() if len(test_data) > 0 else 0
    ax.text(0.02, 0.98, f"Train: {total_train:,} | Test: {total_test:,}",
            transform=ax.transAxes, fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('current_state_models_by_program.png', dpi=150, bbox_inches='tight')
print("  ✓ Saved program-level visualization: current_state_models_by_program.png")
plt.close()

print("\n" + "="*80)
print("HYBRID TRANSITION MODEL BUILD COMPLETE")
print("="*80)
print("\nModel Summary:")
print(f"  Dataset: loan_performance_enhanced.csv")
print(f"  Unique loans: {df['display_id'].nunique():,}")
print(f"  Total transitions analyzed: {len(transitions):,}")
print(f"\n  Regression Models (for CURRENT state):")
print(f"    • Current → D30+ AUC: {auc_d30:.4f}")
print(f"      - Features ({len(feature_cols)}): FICO, amount, term, age, UPB, payments, delinq history, program")
print(f"    • Current → Prepay AUC: {auc_prepay:.4f}")
print(f"      - Features ({len(prepay_feature_cols)}): program, loan_term, loan_age_months ONLY")
print(f"\n  Empirical Matrices:")
print(f"    • Delinquency states covered: {len(transition_matrices)}")
print(f"    • Matrix dimensions: {len(programs)} Programs x {len(term_buckets)} Term buckets")
print(f"\n  Feature Strategy:")
print(f"    • D30+ model: Full feature set with delinquency history")
print(f"    • Prepay model: Simplified - program, term, age only")
print(f"\n  Output Files Generated:")
print(f"    Models:")
print(f"      • hybrid_transition_models.pkl - Model objects and matrices")
print(f"    Feature Importance:")
print(f"      • feature_importance_d30.csv - D30+ feature coefficients")
print(f"      • feature_importance_prepay.csv - Prepay feature coefficients")
print(f"    Predictions:")
print(f"      • current_state_predictions_by_age.csv - Combined D30+ & Prepay predictions by loan age")
print(f"    Visualizations:")
print(f"      • current_state_models_combined.png - Overall side-by-side comparison (1x2)")
print(f"      • current_state_models_by_program.png - Program-level breakdown")
print("="*80)
