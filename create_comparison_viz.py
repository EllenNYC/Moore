#!/usr/bin/env python3
"""
Create comprehensive visualizations comparing modeling approaches
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

print("Creating comparison visualizations...")

# Load results from different approaches
with open('fixed_cashflow_results.pkl', 'rb') as f:
    transition_results = pickle.load(f)

with open('analysis_results_v2.pkl', 'rb') as f:
    calibrated_results = pickle.load(f)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('Set2')

# Create comprehensive figure
fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

# ============================================================================
# 1. IRR Comparison - Unlevered
# ============================================================================
ax1 = fig.add_subplot(gs[0, 0])

scenarios = ['Base Case', 'Moderate Stress', 'Severe Stress']

# Transition model results
trans_unlev_irr = [
    transition_results['results'][s]['unlevered']['irr']*100
    for s in scenarios
]

# Calibrated model results
calib_unlev_irr = [
    calibrated_results['results'][s]['unlevered']['irr']*100
    for s in scenarios
]

x = np.arange(len(scenarios))
width = 0.35

bars1 = ax1.bar(x - width/2, trans_unlev_irr, width, label='Transition Model',
                alpha=0.8, edgecolor='black', color='#2ecc71')
bars2 = ax1.bar(x + width/2, calib_unlev_irr, width, label='Calibrated Model',
                alpha=0.8, edgecolor='black', color='#e74c3c')

ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax1.set_ylabel('Unlevered IRR (%)', fontweight='bold', fontsize=11)
ax1.set_title('Unlevered IRR Comparison', fontweight='bold', fontsize=13)
ax1.set_xticks(x)
ax1.set_xticklabels(scenarios, fontsize=9)
ax1.legend(fontsize=10)
ax1.grid(axis='y', alpha=0.3)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center',
                va='bottom' if height > 0 else 'top', fontsize=9)

# ============================================================================
# 2. IRR Comparison - Levered
# ============================================================================
ax2 = fig.add_subplot(gs[0, 1])

trans_lev_irr = [
    transition_results['results'][s]['levered']['irr']*100
    for s in scenarios
]

calib_lev_irr = [
    calibrated_results['results'][s]['levered']['irr']*100
    for s in scenarios
]

bars1 = ax2.bar(x - width/2, trans_lev_irr, width, label='Transition Model',
                alpha=0.8, edgecolor='black', color='#2ecc71')
bars2 = ax2.bar(x + width/2, calib_lev_irr, width, label='Calibrated Model',
                alpha=0.8, edgecolor='black', color='#e74c3c')

ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax2.set_ylabel('Levered IRR (%)', fontweight='bold', fontsize=11)
ax2.set_title('Levered IRR (85% LTV, 6.5% debt)', fontweight='bold', fontsize=13)
ax2.set_xticks(x)
ax2.set_xticklabels(scenarios, fontsize=9)
ax2.legend(fontsize=10)
ax2.grid(axis='y', alpha=0.3)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center',
                va='bottom' if height > 0 else 'top', fontsize=9)

# ============================================================================
# 3. Loss Rate Comparison
# ============================================================================
ax3 = fig.add_subplot(gs[0, 2])

trans_loss = [
    transition_results['results'][s]['unlevered']['loss_rate']*100
    for s in scenarios
]

calib_loss = [
    calibrated_results['results'][s]['unlevered']['loss_rate']*100
    for s in scenarios
]

bars1 = ax3.bar(x - width/2, trans_loss, width, label='Transition Model',
                alpha=0.8, edgecolor='black', color='#2ecc71')
bars2 = ax3.bar(x + width/2, calib_loss, width, label='Calibrated Model',
                alpha=0.8, edgecolor='black', color='#e74c3c')

ax3.set_ylabel('Loss Rate (%)', fontweight='bold', fontsize=11)
ax3.set_title('Credit Loss Rates', fontweight='bold', fontsize=13)
ax3.set_xticks(x)
ax3.set_xticklabels(scenarios, fontsize=9)
ax3.legend(fontsize=10)
ax3.grid(axis='y', alpha=0.3)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

# ============================================================================
# 4. Monthly Interest Income - Transition Model
# ============================================================================
ax4 = fig.add_subplot(gs[1, :])

for scenario in scenarios:
    cf = transition_results['cashflows'][scenario]
    ax4.plot(cf['month'], cf['interest']/1000, label=f'{scenario} (Transition)',
             linewidth=2.5, marker='o', markersize=4)

ax4.set_xlabel('Month', fontweight='bold', fontsize=11)
ax4.set_ylabel('Monthly Interest Income ($000s)', fontweight='bold', fontsize=11)
ax4.set_title('Monthly Interest Income - Transition Model', fontweight='bold', fontsize=13)
ax4.legend(fontsize=10)
ax4.grid(alpha=0.3)

# ============================================================================
# 5. Cumulative Losses - Both Models
# ============================================================================
ax5 = fig.add_subplot(gs[2, 0])

cf_trans_base = transition_results['cashflows']['Base Case']
cf_calib_base = calibrated_results['cashflows']['Base Case']

ax5.plot(cf_trans_base['month'], cf_trans_base['net_loss'].cumsum()/1000,
         label='Transition Model', linewidth=2.5, color='#2ecc71')
ax5.plot(cf_calib_base['month'], cf_calib_base['net_loss'].cumsum()/1000,
         label='Calibrated Model', linewidth=2.5, linestyle='--', color='#e74c3c')

ax5.set_xlabel('Month', fontweight='bold', fontsize=11)
ax5.set_ylabel('Cumulative Losses ($000s)', fontweight='bold', fontsize=11)
ax5.set_title('Base Case - Cumulative Credit Losses', fontweight='bold', fontsize=13)
ax5.legend(fontsize=10)
ax5.grid(alpha=0.3)

# ============================================================================
# 6. Portfolio Runoff - Transition Model
# ============================================================================
ax6 = fig.add_subplot(gs[2, 1])

for scenario in scenarios:
    cf = transition_results['cashflows'][scenario]
    ax6.plot(cf['month'], cf['ending_balance']/1000000,
             label=scenario, linewidth=2.5)

ax6.set_xlabel('Month', fontweight='bold', fontsize=11)
ax6.set_ylabel('Outstanding Balance ($M)', fontweight='bold', fontsize=11)
ax6.set_title('Portfolio Runoff - Transition Model', fontweight='bold', fontsize=13)
ax6.legend(fontsize=10)
ax6.grid(alpha=0.3)

# ============================================================================
# 7. Delinquency Progression (Transition Model Base Case)
# ============================================================================
ax7 = fig.add_subplot(gs[2, 2])

cf = transition_results['cashflows']['Base Case']
if 'current_count' in cf.columns:
    ax7.plot(cf['month'], cf['current_count'], label='Current', linewidth=2)
    ax7.plot(cf['month'], cf['d30_count'], label='D30-59', linewidth=2)
    ax7.plot(cf['month'], cf['d60_count'], label='D60-89', linewidth=2)
    ax7.plot(cf['month'], cf['d90_count'], label='D90-119', linewidth=2)
    ax7.plot(cf['month'], cf['charged_off_count'], label='Charged-off', linewidth=2, linestyle='--')

    ax7.set_xlabel('Month', fontweight='bold', fontsize=11)
    ax7.set_ylabel('Number of Loans', fontweight='bold', fontsize=11)
    ax7.set_title('Delinquency State Progression (Base)', fontweight='bold', fontsize=13)
    ax7.legend(fontsize=9)
    ax7.grid(alpha=0.3)

# ============================================================================
# 8. MOIC Comparison
# ============================================================================
ax8 = fig.add_subplot(gs[3, 0])

trans_moic = [
    transition_results['results'][s]['unlevered']['moic']
    for s in scenarios
]

calib_moic = [
    calibrated_results['results'][s]['unlevered']['moic']
    for s in scenarios
]

bars1 = ax8.bar(x - width/2, trans_moic, width, label='Transition Model',
                alpha=0.8, edgecolor='black', color='#2ecc71')
bars2 = ax8.bar(x + width/2, calib_moic, width, label='Calibrated Model',
                alpha=0.8, edgecolor='black', color='#e74c3c')

ax8.axhline(y=1.0, color='red', linestyle='--', linewidth=1, label='Break-even')
ax8.set_ylabel('MOIC (x)', fontweight='bold', fontsize=11)
ax8.set_title('Unlevered MOIC Comparison', fontweight='bold', fontsize=13)
ax8.set_xticks(x)
ax8.set_xticklabels(scenarios, fontsize=9)
ax8.legend(fontsize=10)
ax8.grid(axis='y', alpha=0.3)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}x', ha='center', va='bottom', fontsize=9)

# ============================================================================
# 9. Summary Comparison Table
# ============================================================================
ax9 = fig.add_subplot(gs[3, 1:])
ax9.axis('tight')
ax9.axis('off')

table_data = []
for scenario in scenarios:
    trans_u = transition_results['results'][scenario]['unlevered']
    trans_l = transition_results['results'][scenario]['levered']

    table_data.append([
        f"{scenario}\n(Transition)",
        f"{trans_u['irr']*100:.1f}%",
        f"{trans_u['moic']:.2f}x",
        f"{trans_l['irr']*100:.1f}%",
        f"{trans_l['moic']:.2f}x",
        f"{trans_u['loss_rate']*100:.1f}%"
    ])

for scenario in scenarios:
    calib_u = calibrated_results['results'][scenario]['unlevered']
    calib_l = calibrated_results['results'][scenario]['levered']

    table_data.append([
        f"{scenario}\n(Calibrated)",
        f"{calib_u['irr']*100:.1f}%",
        f"{calib_u['moic']:.2f}x",
        f"{calib_l['irr']*100:.1f}%",
        f"{calib_l['moic']:.2f}x",
        f"{calib_u['loss_rate']*100:.1f}%"
    ])

table = ax9.table(cellText=table_data,
                  colLabels=['Scenario/Model', 'Unlev\nIRR', 'Unlev\nMOIC',
                            'Lev\nIRR', 'Lev\nMOIC', 'Loss\nRate'],
                  cellLoc='center',
                  loc='center',
                  colWidths=[0.25, 0.12, 0.12, 0.12, 0.12, 0.12])

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2.2)

# Color header
for i in range(6):
    table[(0, i)].set_facecolor('#3498db')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Color transition model rows
for i in [1, 2, 3]:
    for j in range(6):
        table[(i, j)].set_facecolor('#d5f4e6')

# Color calibrated model rows
for i in [4, 5, 6]:
    for j in range(6):
        table[(i, j)].set_facecolor('#fadbd8')

ax9.set_title('Model Comparison Summary', fontweight='bold', fontsize=13, pad=20)

# Overall title
fig.suptitle('Consumer Credit Portfolio - Model Comparison Analysis\nTransition Model vs Calibrated Historical Model',
             fontsize=16, fontweight='bold', y=0.995)

# Save
plt.savefig('model_comparison_analysis.png', dpi=300, bbox_inches='tight')
print("Saved: model_comparison_analysis.png")

plt.savefig('model_comparison_analysis.pdf', bbox_inches='tight')
print("Saved: model_comparison_analysis.pdf")

# Create transition-specific visualizations
fig2, axes = plt.subplots(2, 2, figsize=(16, 10))
fig2.suptitle('Transition Model - Detailed Analytics', fontsize=16, fontweight='bold')

# Transition roll rates
ax = axes[0, 0]
states = ['CURRENT', 'D1-29', 'D30-59', 'D60-89', 'D90-119', 'D120+']
chargeoff_rates = [0.20, 0.76, 0.90, 1.95, 79.86, 86.36]

bars = ax.bar(states, chargeoff_rates, alpha=0.8, edgecolor='black', color='coral')
ax.set_ylabel('Monthly Charge-off Rate (%)', fontweight='bold')
ax.set_title('Charge-off Rates by Delinquency State', fontweight='bold')
ax.grid(axis='y', alpha=0.3)

for bar, rate in zip(bars, chargeoff_rates):
    ax.text(bar.get_x() + bar.get_width()/2., rate,
            f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')

# Monthly defaults
ax = axes[0, 1]
for scenario in scenarios:
    cf = transition_results['cashflows'][scenario]
    ax.plot(cf['month'], cf['defaults']/1000, label=scenario, linewidth=2.5)

ax.set_xlabel('Month', fontweight='bold')
ax.set_ylabel('Monthly Defaults ($000s)', fontweight='bold')
ax.set_title('Monthly Default Volume', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Monthly prepayments
ax = axes[1, 0]
for scenario in scenarios:
    cf = transition_results['cashflows'][scenario]
    ax.plot(cf['month'], cf['prepayments']/1000, label=scenario, linewidth=2.5)

ax.set_xlabel('Month', fontweight='bold')
ax.set_ylabel('Monthly Prepayments ($000s)', fontweight='bold')
ax.set_title('Monthly Prepayment Volume', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Total inflows
ax = axes[1, 1]
for scenario in scenarios:
    cf = transition_results['cashflows'][scenario]
    ax.plot(cf['month'], cf['total_inflow']/1000, label=scenario, linewidth=2.5)

ax.set_xlabel('Month', fontweight='bold')
ax.set_ylabel('Total Monthly Inflow ($000s)', fontweight='bold')
ax.set_title('Total Monthly Cashflows', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('transition_model_details.png', dpi=300, bbox_inches='tight')
print("Saved: transition_model_details.png")

plt.savefig('transition_model_details.pdf', bbox_inches='tight')
print("Saved: transition_model_details.pdf")

print("\n✓ All visualizations created successfully!")
print("  - model_comparison_analysis.png/pdf")
print("  - transition_model_details.png/pdf")
