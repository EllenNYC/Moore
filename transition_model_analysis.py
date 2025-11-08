#!/usr/bin/env python3
"""
Delinquency Transition Model for Consumer Credit Portfolio
Moore Capital Case Study
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from scipy import optimize
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("DELINQUENCY TRANSITION MODEL ANALYSIS")
print("="*80)

# ============================================================================
# 1. LOAD AND PREPARE DATA
# ============================================================================
print("\n1. Loading and preparing data...")

loan_tape = pd.read_csv('loan tape - moore v1.0.csv')
loan_performance = pd.read_csv('loan performance - moore v1.0.csv')

# Clean loan tape
loan_tape.columns = loan_tape.columns.str.strip()
loan_tape['mdr'] = pd.to_numeric(loan_tape['mdr'].str.rstrip('%'), errors='coerce') / 100
loan_tape['int_rate'] = pd.to_numeric(loan_tape['int_rate'].str.rstrip('%'), errors='coerce') / 100
loan_tape['approved_amount'] = pd.to_numeric(loan_tape['approved_amount'].str.replace('$', '').str.replace(',', ''), errors='coerce')

# Clean performance
loan_performance['report_date'] = pd.to_datetime(loan_performance['report_date'])
loan_performance = loan_performance.sort_values(['display_id', 'report_date'])

print(f"  Loaded {len(loan_tape):,} loans")
print(f"  Performance observations: {len(loan_performance):,}")

# ============================================================================
# 2. CREATE DELINQUENCY STATES
# ============================================================================
print("\n2. Creating delinquency state buckets...")

def assign_state(row):
    """Assign delinquency state based on status and days delinquent"""
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

print("\nState distribution:")
print(loan_performance['state'].value_counts().sort_index())

# ============================================================================
# 3. BUILD TRANSITION DATA
# ============================================================================
print("\n3. Building month-to-month transition data...")

# Create lagged state (previous month's state)
loan_performance['prev_state'] = loan_performance.groupby('display_id')['state'].shift(1)
loan_performance['prev_upb'] = loan_performance.groupby('display_id')['upb'].shift(1)

# Remove first observation for each loan (no previous state)
transitions = loan_performance[loan_performance['prev_state'].notna()].copy()

# Merge with loan characteristics
transitions = transitions.merge(
    loan_tape[['display_id', 'fico_score', 'approved_amount', 'loan_term', 'int_rate', 'mdr', 'program']],
    on='display_id',
    how='left'
)

print(f"  Total transitions observed: {len(transitions):,}")

# Define terminal states
TERMINAL_STATES = ['CHARGED_OFF', 'PAID_OFF']

# Filter to non-terminal previous states for modeling
modeling_transitions = transitions[~transitions['prev_state'].isin(TERMINAL_STATES)].copy()
print(f"  Non-terminal transitions for modeling: {len(modeling_transitions):,}")

# ============================================================================
# 4. BUILD TRANSITION MODELS
# ============================================================================
print("\n4. Building transition probability models...")

# Define transition categories
modeling_transitions['transitioned_to_default'] = (modeling_transitions['state'] == 'CHARGED_OFF').astype(int)
modeling_transitions['transitioned_to_payoff'] = (modeling_transitions['state'] == 'PAID_OFF').astype(int)
modeling_transitions['stayed_or_cured'] = ((modeling_transitions['state'] == 'CURRENT') |
                                            (modeling_transitions['state'] < modeling_transitions['prev_state'])).astype(int)
modeling_transitions['worsened'] = ((modeling_transitions['state'] > modeling_transitions['prev_state']) &
                                     (~modeling_transitions['state'].isin(TERMINAL_STATES))).astype(int)

# Features for modeling
numeric_features = ['fico_score', 'approved_amount', 'loan_term', 'int_rate', 'mdr']

# Add program dummies
program_dummies = pd.get_dummies(modeling_transitions['program'], prefix='program', drop_first=True)

# Add current delinquency state as feature
state_dummies = pd.get_dummies(modeling_transitions['prev_state'], prefix='state', drop_first=True)

# Combine all features
X_all = pd.concat([
    modeling_transitions[numeric_features],
    program_dummies,
    state_dummies
], axis=1)

print(f"\n  Features: {list(X_all.columns)}")

# ============================================================================
# Model 1: Transition to CHARGED_OFF
# ============================================================================
print("\n  Model 1: Transition to Charge-off...")

y_chargeoff = modeling_transitions['transitioned_to_default']

# Remove missing values
valid_idx = X_all.notna().all(axis=1) & y_chargeoff.notna()
X_clean = X_all[valid_idx]
y_clean = y_chargeoff[valid_idx]

if y_clean.sum() > 100:  # Need enough positive examples
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean, test_size=0.3, random_state=42, stratify=y_clean
    )

    scaler_chargeoff = StandardScaler()
    X_train_scaled = scaler_chargeoff.fit_transform(X_train)
    X_test_scaled = scaler_chargeoff.transform(X_test)

    model_chargeoff = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    model_chargeoff.fit(X_train_scaled, y_train)

    auc_chargeoff = roc_auc_score(y_test, model_chargeoff.predict_proba(X_test_scaled)[:, 1])
    print(f"    AUC: {auc_chargeoff:.4f}")
    print(f"    Charge-off rate in training: {y_train.mean()*100:.2f}%")
else:
    print(f"    Insufficient charge-off events for modeling")
    model_chargeoff = None
    scaler_chargeoff = None

# ============================================================================
# Model 2: Transition to PAID_OFF
# ============================================================================
print("\n  Model 2: Transition to Paid-off...")

y_payoff = modeling_transitions['transitioned_to_payoff']

valid_idx = X_all.notna().all(axis=1) & y_payoff.notna()
X_clean = X_all[valid_idx]
y_clean = y_payoff[valid_idx]

if y_clean.sum() > 100:
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean, test_size=0.3, random_state=42, stratify=y_clean
    )

    scaler_payoff = StandardScaler()
    X_train_scaled = scaler_payoff.fit_transform(X_train)
    X_test_scaled = scaler_payoff.transform(X_test)

    model_payoff = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    model_payoff.fit(X_train_scaled, y_train)

    auc_payoff = roc_auc_score(y_test, model_payoff.predict_proba(X_test_scaled)[:, 1])
    print(f"    AUC: {auc_payoff:.4f}")
    print(f"    Payoff rate in training: {y_train.mean()*100:.2f}%")
else:
    print(f"    Insufficient payoff events for modeling")
    model_payoff = None
    scaler_payoff = None

# ============================================================================
# Model 3: Delinquency Worsening (roll rate)
# ============================================================================
print("\n  Model 3: Delinquency worsening (roll rate)...")

y_worsen = modeling_transitions['worsened']

valid_idx = X_all.notna().all(axis=1) & y_worsen.notna()
X_clean = X_all[valid_idx]
y_clean = y_worsen[valid_idx]

if y_clean.sum() > 100:
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean, test_size=0.3, random_state=42, stratify=y_clean
    )

    scaler_worsen = StandardScaler()
    X_train_scaled = scaler_worsen.fit_transform(X_train)
    X_test_scaled = scaler_worsen.transform(X_test)

    model_worsen = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    model_worsen.fit(X_train_scaled, y_train)

    auc_worsen = roc_auc_score(y_test, model_worsen.predict_proba(X_test_scaled)[:, 1])
    print(f"    AUC: {auc_worsen:.4f}")
    print(f"    Worsening rate in training: {y_train.mean()*100:.2f}%")
else:
    print(f"    Insufficient worsening events for modeling")
    model_worsen = None
    scaler_worsen = None

# ============================================================================
# 5. COMPUTE EMPIRICAL TRANSITION MATRIX
# ============================================================================
print("\n5. Computing empirical transition matrix...")

states = ['CURRENT', 'D1_29', 'D30_59', 'D60_89', 'D90_119', 'D120_PLUS', 'CHARGED_OFF', 'PAID_OFF']

transition_matrix = pd.DataFrame(0, index=states, columns=states)

for from_state in states:
    if from_state in TERMINAL_STATES:
        transition_matrix.loc[from_state, from_state] = 1.0  # Terminal states stay
    else:
        state_transitions = transitions[transitions['prev_state'] == from_state]
        if len(state_transitions) > 0:
            for to_state in states:
                count = (state_transitions['state'] == to_state).sum()
                transition_matrix.loc[from_state, to_state] = count / len(state_transitions)

print("\nEmpirical Transition Matrix (monthly):")
print(transition_matrix.round(3).to_string())

# Calculate roll rates
print("\n" + "="*80)
print("KEY ROLL RATES (from each bucket)")
print("="*80)

for from_state in ['CURRENT', 'D30_59', 'D60_89', 'D90_119']:
    if from_state in transition_matrix.index:
        to_chargeoff = transition_matrix.loc[from_state, 'CHARGED_OFF']
        to_payoff = transition_matrix.loc[from_state, 'PAID_OFF']
        to_current = transition_matrix.loc[from_state, 'CURRENT']

        print(f"\nFrom {from_state}:")
        print(f"  → Charge-off: {to_chargeoff*100:.2f}%")
        print(f"  → Paid-off: {to_payoff*100:.2f}%")
        print(f"  → Current (cure): {to_current*100:.2f}%")

# ============================================================================
# 6. SAVE MODELS
# ============================================================================
print("\n6. Saving transition models...")

import pickle

models_data = {
    'model_chargeoff': model_chargeoff,
    'scaler_chargeoff': scaler_chargeoff,
    'model_payoff': model_payoff,
    'scaler_payoff': scaler_payoff,
    'model_worsen': model_worsen,
    'scaler_worsen': scaler_worsen,
    'feature_cols': list(X_all.columns),
    'transition_matrix': transition_matrix,
    'states': states,
    'numeric_features': numeric_features
}

with open('transition_models.pkl', 'wb') as f:
    pickle.dump(models_data, f)

print("  Transition models saved to transition_models.pkl")

print("\n" + "="*80)
print("TRANSITION MODEL BUILD COMPLETE")
print("="*80)
