#!/usr/bin/env python3
"""
Test adding 'program' feature to regression models
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("TESTING MODELS WITH 'PROGRAM' FEATURE ADDED")
print("="*80)

# Load and clean data
loan_tape = pd.read_csv('loan tape - moore v1.0.csv')
loan_performance = pd.read_csv('loan performance - moore v1.0.csv')

loan_tape.columns = loan_tape.columns.str.strip()
loan_tape['mdr'] = pd.to_numeric(loan_tape['mdr'].str.rstrip('%'), errors='coerce') / 100
loan_tape['int_rate'] = pd.to_numeric(loan_tape['int_rate'].str.rstrip('%'), errors='coerce') / 100
loan_tape['approved_amount'] = pd.to_numeric(loan_tape['approved_amount'].str.replace('$', '').str.replace(',', ''), errors='coerce')

# Get final outcomes
loan_final = loan_performance.sort_values('report_date').groupby('display_id').last().reset_index()
loan_analysis = loan_tape.merge(loan_final[['display_id', 'loan_status']], on='display_id', how='left')
loan_analysis['defaulted'] = loan_analysis['loan_status'].isin(['CHARGED_OFF', 'WRITTEN_OFF']).astype(int)
loan_analysis['prepaid'] = loan_analysis['loan_status'].isin(['PAID_OFF', 'SATISFIED']).astype(int)

print("\nProgram distribution:")
print(loan_analysis['program'].value_counts())

# ============================================================================
# DEFAULT MODEL WITH PROGRAM
# ============================================================================
print("\n" + "="*80)
print("DEFAULT PROBABILITY MODEL")
print("="*80)

# Prepare data
modeling_data = loan_analysis[loan_analysis['loan_status'] != 'CURRENT'].copy()

# Original features
numeric_features = ['fico_score', 'approved_amount', 'loan_term', 'int_rate', 'mdr']

# Create one-hot encoded program variable
program_dummies = pd.get_dummies(modeling_data['program'], prefix='program', drop_first=True)

# Combine features
X_with_program = pd.concat([
    modeling_data[numeric_features],
    program_dummies
], axis=1)

y = modeling_data['defaulted']

# Remove rows with missing values
valid_idx = X_with_program.notna().all(axis=1) & y.notna()
X_clean = X_with_program[valid_idx]
y_clean = y[valid_idx]

print(f"\nDataset size: {len(X_clean):,} loans")
print(f"Features: {list(X_clean.columns)}")
print(f"Default rate: {y_clean.mean() * 100:.2f}%")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_clean, y_clean, test_size=0.3, random_state=42, stratify=y_clean
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
default_model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
default_model.fit(X_train_scaled, y_train)

# Evaluate
y_pred_proba_test = default_model.predict_proba(X_test_scaled)[:, 1]
test_auc = roc_auc_score(y_test, y_pred_proba_test)

print(f"\n{'='*80}")
print("RESULTS - Default Model")
print("="*80)
print(f"\nTest AUC: {test_auc:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X_clean.columns,
    'Coefficient': default_model.coef_[0],
    'Abs_Coefficient': np.abs(default_model.coef_[0])
}).sort_values('Abs_Coefficient', ascending=False)

print("\nFeature Importance (sorted by magnitude):")
print(feature_importance.to_string(index=False))

# ============================================================================
# PREPAYMENT MODEL WITH PROGRAM
# ============================================================================
print("\n" + "="*80)
print("PREPAYMENT PROBABILITY MODEL")
print("="*80)

# Prepare data (only non-defaulted loans)
prepay_data = modeling_data[modeling_data['defaulted'] == 0].copy()

# Create one-hot encoded program variable
program_dummies_prepay = pd.get_dummies(prepay_data['program'], prefix='program', drop_first=True)

# Combine features
X_prepay_with_program = pd.concat([
    prepay_data[numeric_features],
    program_dummies_prepay
], axis=1)

y_prepay = prepay_data['prepaid']

# Remove rows with missing values
valid_idx_p = X_prepay_with_program.notna().all(axis=1) & y_prepay.notna()
X_prepay_clean = X_prepay_with_program[valid_idx_p]
y_prepay_clean = y_prepay[valid_idx_p]

print(f"\nDataset size: {len(X_prepay_clean):,} loans")
print(f"Prepayment rate: {y_prepay_clean.mean() * 100:.2f}%")

# Split data
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
    X_prepay_clean, y_prepay_clean, test_size=0.3, random_state=42, stratify=y_prepay_clean
)

# Scale features
scaler_prepay = StandardScaler()
X_train_p_scaled = scaler_prepay.fit_transform(X_train_p)
X_test_p_scaled = scaler_prepay.transform(X_test_p)

# Train model
prepay_model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
prepay_model.fit(X_train_p_scaled, y_train_p)

# Evaluate
y_pred_proba_test_p = prepay_model.predict_proba(X_test_p_scaled)[:, 1]
test_auc_p = roc_auc_score(y_test_p, y_pred_proba_test_p)

print(f"\n{'='*80}")
print("RESULTS - Prepayment Model")
print("="*80)
print(f"\nTest AUC: {test_auc_p:.4f}")

# Feature importance
feature_importance_p = pd.DataFrame({
    'Feature': X_prepay_clean.columns,
    'Coefficient': prepay_model.coef_[0],
    'Abs_Coefficient': np.abs(prepay_model.coef_[0])
}).sort_values('Abs_Coefficient', ascending=False)

print("\nFeature Importance (sorted by magnitude):")
print(feature_importance_p.to_string(index=False))

# ============================================================================
# COMPARISON
# ============================================================================
print("\n" + "="*80)
print("COMPARISON: ORIGINAL vs WITH PROGRAM")
print("="*80)

print("\nOriginal model performance (from earlier run):")
print("  Default model AUC: 0.8494")
print("  Prepayment model AUC: 0.7146")

print(f"\nNew model performance (with program feature):")
print(f"  Default model AUC: {test_auc:.4f}")
print(f"  Prepayment model AUC: {test_auc_p:.4f}")

print(f"\nImprovement:")
print(f"  Default model: {(test_auc - 0.8494)*100:+.2f} percentage points")
print(f"  Prepayment model: {(test_auc_p - 0.7146)*100:+.2f} percentage points")

# Check statistical significance of program coefficients
print("\n" + "="*80)
print("PROGRAM FEATURE ANALYSIS")
print("="*80)

program_features_default = [col for col in feature_importance['Feature'] if col.startswith('program_')]
program_importance_default = feature_importance[feature_importance['Feature'].isin(program_features_default)]

print("\nDefault Model - Program coefficients:")
print(program_importance_default.to_string(index=False))

program_features_prepay = [col for col in feature_importance_p['Feature'] if col.startswith('program_')]
program_importance_prepay = feature_importance_p[feature_importance_p['Feature'].isin(program_features_prepay)]

print("\nPrepayment Model - Program coefficients:")
print(program_importance_prepay.to_string(index=False))

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

if test_auc > 0.8494:
    print("\n✓ Adding 'program' IMPROVED the default model")
else:
    print("\n✗ Adding 'program' did NOT improve the default model")

if test_auc_p > 0.7146:
    print("✓ Adding 'program' IMPROVED the prepayment model")
else:
    print("✗ Adding 'program' did NOT improve the prepayment model")

print("\n" + "="*80)
