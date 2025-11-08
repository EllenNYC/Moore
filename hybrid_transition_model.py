#!/usr/bin/env python3
"""
Hybrid Transition Model for Consumer Credit Portfolio
- Regression for Current → D30 and Current → Prepay
- Empirical matrices (FICO x Age) for all other transitions
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("HYBRID TRANSITION MODEL - REGRESSION + EMPIRICAL MATRICES")
print("="*80)

# ============================================================================
# 1. LOAD AND PREPARE DATA
# ============================================================================
print("\n1. Loading data...")

loan_tape = pd.read_csv('loan tape - moore v1.0.csv')
loan_performance = pd.read_csv('loan performance - moore v1.0.csv')

# Clean
loan_tape.columns = loan_tape.columns.str.strip()
loan_tape['mdr'] = pd.to_numeric(loan_tape['mdr'].str.rstrip('%'), errors='coerce') / 100
loan_tape['int_rate'] = pd.to_numeric(loan_tape['int_rate'].str.rstrip('%'), errors='coerce') / 100
loan_tape['approved_amount'] = pd.to_numeric(loan_tape['approved_amount'].str.replace('$', '').str.replace(',', ''), errors='coerce')
loan_tape['disbursement_date'] = pd.to_datetime(loan_tape['disbursement_d'], errors='coerce')

loan_performance['report_date'] = pd.to_datetime(loan_performance['report_date'])
loan_performance = loan_performance.sort_values(['display_id', 'report_date'])

print(f"  Loaded {len(loan_tape):,} loans, {len(loan_performance):,} observations")

# ============================================================================
# 2. CREATE STATES AND TRANSITIONS
# ============================================================================
print("\n2. Creating delinquency states...")

def assign_state(row):
    """Assign delinquency state"""
    if row['loan_status'] in ['CHARGED_OFF', 'WRITTEN_OFF']:
        return 'CHARGED_OFF'
    elif row['loan_status'] in ['PAID_OFF', 'SATISFIED']:
        return 'PAID_OFF'
    elif row['days_delinquent'] == 0:
        return 'CURRENT'
    elif row['days_delinquent'] < 30:
        return 'D1_29'
    elif row['days_delinquent'] < 60:
        return 'D30_59'
    elif row['days_delinquent'] < 90:
        return 'D60_89'
    elif row['days_delinquent'] < 120:
        return 'D90_119'
    else:
        return 'D120_PLUS'

loan_performance['state'] = loan_performance.apply(assign_state, axis=1)

# Calculate loan age in months
loan_tape_with_date = loan_tape[['display_id', 'disbursement_date', 'fico_score']].dropna()
loan_performance = loan_performance.merge(loan_tape_with_date, on='display_id', how='left')
loan_performance['loan_age_days'] = (loan_performance['report_date'] - loan_performance['disbursement_date']).dt.days
loan_performance['loan_age_months'] = (loan_performance['loan_age_days'] / 30.44).fillna(0).astype(int)

# Create lagged state
loan_performance['prev_state'] = loan_performance.groupby('display_id')['state'].shift(1)

# Remove first obs
transitions = loan_performance[loan_performance['prev_state'].notna()].copy()

# Merge full loan characteristics
transitions = transitions.merge(
    loan_tape[['display_id', 'approved_amount', 'loan_term', 'int_rate', 'mdr', 'program']],
    on='display_id',
    how='left'
)

print(f"  Total transitions: {len(transitions):,}")

# ============================================================================
# 3. BUILD REGRESSION MODELS (CURRENT → D30 AND CURRENT → PREPAY)
# ============================================================================
print("\n3. Building regression models for Current state transitions...")

# Filter to transitions from CURRENT state only
current_transitions = transitions[transitions['prev_state'] == 'CURRENT'].copy()

print(f"  Current-state transitions: {len(current_transitions):,}")
print(f"  Distribution:")
print(current_transitions['state'].value_counts())

# Create outcome variables
current_transitions['to_d30'] = current_transitions['state'].isin(['D1_29', 'D30_59', 'D60_89', 'D90_119', 'D120_PLUS']).astype(int)
current_transitions['to_prepay'] = (current_transitions['state'] == 'PAID_OFF').astype(int)
current_transitions['to_chargeoff'] = (current_transitions['state'] == 'CHARGED_OFF').astype(int)

print(f"\n  To delinquency (D30+): {current_transitions['to_d30'].mean()*100:.2f}%")
print(f"  To prepay: {current_transitions['to_prepay'].mean()*100:.2f}%")
print(f"  To charge-off: {current_transitions['to_chargeoff'].mean()*100:.2f}%")

# Features
numeric_features = ['fico_score', 'approved_amount', 'loan_term', 'int_rate', 'mdr', 'loan_age_months']
program_dummies = pd.get_dummies(current_transitions['program'], prefix='program', drop_first=True)
X = pd.concat([current_transitions[numeric_features], program_dummies], axis=1)

# ============================================================================
# Model A: Current → D30+ (First Delinquency)
# ============================================================================
print("\n  Model A: Current → D30+ (First Delinquency)...")

y_d30 = current_transitions['to_d30']
valid_idx = X.notna().all(axis=1) & y_d30.notna()
X_clean = X[valid_idx]
y_clean = y_d30[valid_idx]

X_train, X_test, y_train, y_test = train_test_split(
    X_clean, y_clean, test_size=0.3, random_state=42, stratify=y_clean
)

scaler_d30 = StandardScaler()
X_train_scaled = scaler_d30.fit_transform(X_train)
X_test_scaled = scaler_d30.transform(X_test)

model_d30 = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
model_d30.fit(X_train_scaled, y_train)

auc_d30 = roc_auc_score(y_test, model_d30.predict_proba(X_test_scaled)[:, 1])
print(f"    AUC: {auc_d30:.4f}")

# Feature importance
feature_cols = list(X_clean.columns)
coefs_d30 = pd.DataFrame({
    'Feature': feature_cols,
    'Coefficient': model_d30.coef_[0]
}).sort_values('Coefficient', key=abs, ascending=False)
print(f"    Top 3 features: {', '.join(coefs_d30.head(3)['Feature'].tolist())}")

# ============================================================================
# Model B: Current → Prepaid
# ============================================================================
print("\n  Model B: Current → Prepaid...")

y_prepay = current_transitions['to_prepay']
valid_idx = X.notna().all(axis=1) & y_prepay.notna()
X_clean = X[valid_idx]
y_clean = y_prepay[valid_idx]

X_train, X_test, y_train, y_test = train_test_split(
    X_clean, y_clean, test_size=0.3, random_state=42, stratify=y_clean
)

scaler_prepay = StandardScaler()
X_train_scaled = scaler_prepay.fit_transform(X_train)
X_test_scaled = scaler_prepay.transform(X_test)

model_prepay = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
model_prepay.fit(X_train_scaled, y_train)

auc_prepay = roc_auc_score(y_test, model_prepay.predict_proba(X_test_scaled)[:, 1])
print(f"    AUC: {auc_prepay:.4f}")

coefs_prepay = pd.DataFrame({
    'Feature': feature_cols,
    'Coefficient': model_prepay.coef_[0]
}).sort_values('Coefficient', key=abs, ascending=False)
print(f"    Top 3 features: {', '.join(coefs_prepay.head(3)['Feature'].tolist())}")

# ============================================================================
# 4. BUILD EMPIRICAL MATRICES (FICO x AGE) FOR OTHER TRANSITIONS
# ============================================================================
print("\n4. Building empirical transition matrices (FICO x Loan Age)...")

# Define FICO buckets
fico_bins = [0, 600, 650, 700, 750, 900]
fico_labels = ['<600', '600-650', '650-700', '700-750', '750+']
transitions['fico_bucket'] = pd.cut(transitions['fico_score'], bins=fico_bins, labels=fico_labels)

# Define loan age buckets
age_bins = [0, 3, 6, 12, 24, 999]
age_labels = ['0-3m', '3-6m', '6-12m', '12-24m', '24m+']
transitions['age_bucket'] = pd.cut(transitions['loan_age_months'], bins=age_bins, labels=age_labels, right=False)

# Define transition states (excluding CURRENT which uses regression)
delinq_states = ['D1_29', 'D30_59', 'D60_89', 'D90_119', 'D120_PLUS']
all_states = ['CURRENT'] + delinq_states + ['CHARGED_OFF', 'PAID_OFF']

# Build matrices for each delinquency state
transition_matrices = {}

for from_state in delinq_states:
    print(f"\n  Building matrix for {from_state}...")

    state_trans = transitions[transitions['prev_state'] == from_state].copy()

    if len(state_trans) < 100:
        print(f"    Insufficient data ({len(state_trans)} obs), using overall average")
        continue

    # Create matrix: rows = FICO buckets, columns = Age buckets
    matrix_dict = {}

    for fico in fico_labels:
        for age in age_labels:
            subset = state_trans[(state_trans['fico_bucket'] == fico) &
                                 (state_trans['age_bucket'] == age)]

            if len(subset) >= 10:  # Minimum observations
                # Calculate transition probabilities
                trans_probs = {}
                for to_state in all_states:
                    trans_probs[to_state] = (subset['state'] == to_state).mean()

                matrix_dict[(fico, age)] = trans_probs
            else:
                # Fall back to FICO bucket average
                fico_subset = state_trans[state_trans['fico_bucket'] == fico]
                if len(fico_subset) >= 10:
                    trans_probs = {}
                    for to_state in all_states:
                        trans_probs[to_state] = (fico_subset['state'] == to_state).mean()
                    matrix_dict[(fico, age)] = trans_probs
                else:
                    # Fall back to overall average for this state
                    trans_probs = {}
                    for to_state in all_states:
                        trans_probs[to_state] = (state_trans['state'] == to_state).mean()
                    matrix_dict[(fico, age)] = trans_probs

    transition_matrices[from_state] = matrix_dict

    # Print sample rates
    sample_key = list(matrix_dict.keys())[0]
    sample_probs = matrix_dict[sample_key]
    print(f"    Sample (FICO={sample_key[0]}, Age={sample_key[1]}):")
    print(f"      → Charged-off: {sample_probs.get('CHARGED_OFF', 0)*100:.2f}%")
    print(f"      → Paid-off: {sample_probs.get('PAID_OFF', 0)*100:.2f}%")

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

        print(f"\n  {from_state}:")
        print(f"    → Charge-off: {to_co*100:.2f}%")
        print(f"    → Paid-off: {to_po*100:.2f}%")
        print(f"    → Current (cure): {to_curr*100:.2f}%")
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
    'feature_cols': feature_cols,
    'numeric_features': numeric_features,

    # Empirical matrices (for all delinquency states)
    'transition_matrices': transition_matrices,
    'fico_bins': fico_bins,
    'fico_labels': fico_labels,
    'age_bins': age_bins,
    'age_labels': age_labels,

    # States
    'all_states': all_states,
    'delinq_states': delinq_states,

    # Model performance
    'auc_d30': auc_d30,
    'auc_prepay': auc_prepay
}

with open('hybrid_transition_models.pkl', 'wb') as f:
    pickle.dump(hybrid_models, f)

print("  Saved to hybrid_transition_models.pkl")

print("\n" + "="*80)
print("HYBRID TRANSITION MODEL BUILD COMPLETE")
print("="*80)
print("\nModel Summary:")
print(f"  • Current → D30 Regression AUC: {auc_d30:.4f}")
print(f"  • Current → Prepay Regression AUC: {auc_prepay:.4f}")
print(f"  • Empirical matrices built for {len(transition_matrices)} delinquency states")
print(f"  • Matrix dimensions: {len(fico_labels)} FICO buckets x {len(age_labels)} age buckets")
