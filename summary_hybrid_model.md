# Hybrid Transition Model Summary

## Model Structure

### 1. Regression Models (for CURRENT loans only)
- **Current → D30+** (First Delinquency)
  - AUC: 0.7484
  - Top features: FICO score, MDR, Loan age
  
- **Current → Prepaid**
  - AUC: 0.7233
  - Top features: Loan term, Interest rate, MDR

### 2. Empirical Matrices (FICO x Loan Age)
Built for delinquent states: D1-29, D30-59, D60-89, D90-119, D120+

Matrix dimensions: 5 FICO buckets x 5 age buckets

### Roll Rates by State

| State | → Charge-off | → Paid-off | → Current (Cure) | Observations |
|-------|--------------|------------|------------------|--------------|
| **CURRENT** | 0.20% | 6.40% | 88.63% | 510,778 |
| **D1-29** | 0.76% | 5.07% | 26.16% | 35,105 |
| **D30-59** | 0.90% | 3.21% | 6.97% | 13,475 |
| **D60-89** | 1.95% | 1.82% | 2.53% | 9,797 |
| **D90-119** | **79.86%** | 1.32% | 0.78% | 7,811 |
| **D120+** | **86.36%** | 1.73% | 0.48% | 1,041 |

## Key Insights

1. **First Delinquency** driven by borrower characteristics (FICO, age, program)
2. **Later transitions** follow empirical patterns based on FICO bucket and loan age
3. **Critical threshold at D90**: Once loans reach 90+ days delinquent, 80% monthly charge-off rate
4. **Cure rates decline** as delinquency deepens (26% from D1-29 down to 0.8% from D90+)

## Files Created

1. `hybrid_transition_model.py` - Model building code
2. `cashflow_hybrid_model.py` - Cashflow projection (needs debugging)
3. `hybrid_transition_models.pkl` - Saved models
4. `hybrid_cashflow_results.pkl` - Results (currently showing anomalous 0% losses)

## Next Steps

The hybrid approach is conceptually correct but the cashflow projection has a bug preventing transitions from being applied correctly. The issue appears to be in how state transitions are being simulated and recorded in the monthly cashflow loop.
