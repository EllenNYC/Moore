#!/usr/bin/env python3
"""
Create visualizations for Moore Capital investment memo
Updated to use hybrid_cashflow_results.pkl from hybrid transition model
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load results from hybrid cashflow model
print("Loading hybrid cashflow results...")
with open('hybrid_cashflow_results.pkl', 'rb') as f:
    data = pickle.load(f)

results = data['results']
cashflows = data['cashflows']
print(f"Loaded {len(results)} scenarios")

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('Set2')

# Create comprehensive visualization
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. IRR Comparison
ax1 = fig.add_subplot(gs[0, 0])
scenarios = ['Base Case', 'Moderate Stress', 'Severe Stress']
unlev_irr = [results[s]['unlevered']['irr']*100 for s in scenarios]
lev_irr = [results[s]['levered']['irr']*100 for s in scenarios]

x = np.arange(len(scenarios))
width = 0.35
bars1 = ax1.bar(x - width/2, unlev_irr, width, label='Unlevered', alpha=0.8, edgecolor='black')
bars2 = ax1.bar(x + width/2, lev_irr, width, label='Levered (85% LTV)', alpha=0.8, edgecolor='black')
ax1.axhline(y=0, color='red', linestyle='--', linewidth=1)
ax1.set_ylabel('IRR (%)', fontweight='bold', fontsize=11)
ax1.set_title('Returns Across Scenarios', fontweight='bold', fontsize=12)
ax1.set_xticks(x)
ax1.set_xticklabels(scenarios, fontsize=9)
ax1.legend(fontsize=9)
ax1.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom' if height > 0 else 'top', fontsize=8)

# 2. MOIC Comparison
ax2 = fig.add_subplot(gs[0, 1])
unlev_moic = [results[s]['unlevered']['moic'] for s in scenarios]
lev_moic = [results[s]['levered']['moic'] for s in scenarios]

bars1 = ax2.bar(x - width/2, unlev_moic, width, label='Unlevered', alpha=0.8, edgecolor='black')
bars2 = ax2.bar(x + width/2, lev_moic, width, label='Levered (85% LTV)', alpha=0.8, edgecolor='black')
ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=1, label='Break-even')
ax2.set_ylabel('MOIC (x)', fontweight='bold', fontsize=11)
ax2.set_title('Multiple on Invested Capital', fontweight='bold', fontsize=12)
ax2.set_xticks(x)
ax2.set_xticklabels(scenarios, fontsize=9)
ax2.legend(fontsize=9)
ax2.grid(axis='y', alpha=0.3)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}x', ha='center', va='bottom' if height > 0 else 'top', fontsize=8)

# 3. Loss Rates
ax3 = fig.add_subplot(gs[0, 2])
loss_rates = [results[s]['unlevered']['loss_rate']*100 for s in scenarios]
bars = ax3.bar(scenarios, loss_rates, alpha=0.8, edgecolor='black', color='coral')
ax3.set_ylabel('Loss Rate (%)', fontweight='bold', fontsize=11)
ax3.set_title('Credit Loss Rates by Scenario', fontweight='bold', fontsize=12)
ax3.set_xticklabels(scenarios, fontsize=9)
ax3.grid(axis='y', alpha=0.3)

for i, bar in enumerate(bars):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%', ha='center', va='bottom', fontsize=8)

# 4. Monthly Interest Income
ax4 = fig.add_subplot(gs[1, :])
for scenario in scenarios:
    cf = cashflows[scenario]
    ax4.plot(cf['month'], cf['interest'], label=scenario, linewidth=2, marker='o', markersize=3)
ax4.set_xlabel('Month', fontweight='bold', fontsize=11)
ax4.set_ylabel('Monthly Interest Income ($)', fontweight='bold', fontsize=11)
ax4.set_title('Projected Monthly Interest Income', fontweight='bold', fontsize=12)
ax4.legend(fontsize=10)
ax4.grid(alpha=0.3)

# 5. Cumulative Losses
ax5 = fig.add_subplot(gs[2, 0])
for scenario in scenarios:
    cf = cashflows[scenario]
    ax5.plot(cf['month'], cf['net_loss'].cumsum(), label=scenario, linewidth=2)
ax5.set_xlabel('Month', fontweight='bold', fontsize=11)
ax5.set_ylabel('Cumulative Net Losses ($)', fontweight='bold', fontsize=11)
ax5.set_title('Credit Losses Over Time', fontweight='bold', fontsize=12)
ax5.legend(fontsize=10)
ax5.grid(alpha=0.3)

# 6. Portfolio Runoff
ax6 = fig.add_subplot(gs[2, 1])
for scenario in scenarios:
    cf = cashflows[scenario]
    ax6.plot(cf['month'], cf['ending_balance']/1000000, label=scenario, linewidth=2)
ax6.set_xlabel('Month', fontweight='bold', fontsize=11)
ax6.set_ylabel('Outstanding Balance ($M)', fontweight='bold', fontsize=11)
ax6.set_title('Portfolio Runoff Profile', fontweight='bold', fontsize=12)
ax6.legend(fontsize=10)
ax6.grid(alpha=0.3)

# 7. Summary Metrics Table
ax7 = fig.add_subplot(gs[2, 2])
ax7.axis('tight')
ax7.axis('off')

table_data = []
for scenario in scenarios:
    unlev = results[scenario]['unlevered']
    table_data.append([
        scenario,
        f"{unlev['irr']*100:.1f}%",
        f"{unlev['moic']:.2f}x",
        f"{unlev['loss_rate']*100:.1f}%",
        f"{unlev['wal_years']:.1f}y"
    ])

table = ax7.table(cellText=table_data,
                  colLabels=['Scenario', 'IRR', 'MOIC', 'Loss%', 'WAL'],
                  cellLoc='center',
                  loc='center',
                  colWidths=[0.3, 0.15, 0.15, 0.15, 0.15])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Style header
for i in range(5):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

ax7.set_title('Unlevered Returns Summary', fontweight='bold', fontsize=12, pad=20)

fig.suptitle('Moore Capital Consumer Credit Portfolio Analysis - Hybrid Transition Model',
             fontsize=16, fontweight='bold', y=0.995)

plt.savefig('investment_analysis_charts.png', dpi=300, bbox_inches='tight')
print("Saved investment_analysis_charts.png")

plt.savefig('investment_analysis_charts.pdf', bbox_inches='tight')
print("Saved investment_analysis_charts.pdf")

plt.show()
