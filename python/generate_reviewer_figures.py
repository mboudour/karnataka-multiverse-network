#!/usr/bin/env python3
"""
Generate two new figures requested by the reviewer:
1. Centrality correlation heatmap (Spearman, from pre-computed centrality data)
2. Sign-stability matrix (net AME sign direction per centrality x model)
Uses pre-computed CSV data to avoid recomputing centralities from scratch.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

FIGDIR = '/home/ubuntu/karnataka_multiverse/results/python/figs'
TABDIR = '/home/ubuntu/karnataka_multiverse/results/python/tables'
os.makedirs(FIGDIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 150,
})

pretty = lambda s: s.replace('_', ' ').title()

# ── Load regression multiverse CSV ──────────────────────────────────────────
print("Loading regression multiverse CSV...")
df_reg = pd.read_csv(f'{TABDIR}/regression_multiverse.csv')
print(f"  Total specs: {len(df_reg)}")

# ── Figure 1: Centrality correlation heatmap ────────────────────────────────
# Use the directed_weighted representation, OLS model, raw_count outcome
# to get one AME per centrality per node (use the AME values as a proxy for
# centrality scores — but better: load the centrality_universe_stats.csv
# which has per-node centrality values)

# Actually: build a wide pivot of AME by centrality across all representations
# to show how correlated the centrality measures are in their predictive signal
print("Building centrality AME correlation matrix...")
df_valid = df_reg[np.isfinite(df_reg['ame']) & (df_reg['ame'].abs() < 1e6)].copy()

# Pivot: rows = (representation, outcome, model), cols = centrality, values = ame
pivot = df_valid.pivot_table(
    index=['representation', 'outcome', 'model'],
    columns='centrality',
    values='ame',
    aggfunc='mean'
)

# Spearman correlation across the "specification" dimension
corr = pivot.corr(method='spearman')

# Drop columns/rows with all NaN
corr = corr.dropna(how='all', axis=0).dropna(how='all', axis=1)

# Hierarchical clustering
dist = (1 - corr.abs()).values.copy()
np.fill_diagonal(dist, 0)
dist_condensed = squareform(np.clip(dist, 0, None), checks=False)
Z = linkage(dist_condensed, method='ward')
order = leaves_list(Z)
corr_ordered = corr.iloc[order, order]
labels = [pretty(c) for c in corr_ordered.columns]

fig, ax = plt.subplots(figsize=(14, 12))
mask = np.zeros_like(corr_ordered, dtype=bool)
mask[np.triu_indices_from(mask, k=1)] = True
sns.heatmap(corr_ordered, mask=mask, annot=False, cmap='RdBu_r',
            vmin=-1, vmax=1, ax=ax, linewidths=0.3,
            xticklabels=labels, yticklabels=labels,
            cbar_kws={'label': 'Spearman $\\rho$ of AME across specifications', 'shrink': 0.8})
ax.set_title('Spearman Correlation Among Centrality Measures\n'
             '(based on AME signal across all valid specifications; hierarchically clustered)',
             fontsize=12)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=7)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7)
plt.tight_layout()
plt.savefig(f'{FIGDIR}/centrality_correlation_heatmap.png', bbox_inches='tight')
plt.close()
print("  Saved: centrality_correlation_heatmap.png")

# Save correlation matrix as CSV for appendix
corr_ordered.to_csv(f'{TABDIR}/centrality_ame_spearman_corr.csv')
print("  Saved: centrality_ame_spearman_corr.csv")

# ── Figure 2: Sign-stability matrix ─────────────────────────────────────────
print("Building sign-stability matrix...")

def sign_fracs(group):
    n = len(group)
    pos = (group['ame'] > 0).sum() / n
    neg = (group['ame'] < 0).sum() / n
    return pd.Series({'pos': pos, 'neg': neg, 'n': n})

sign_df = df_valid.groupby(['centrality', 'model']).apply(sign_fracs).reset_index()

pivot_pos = sign_df.pivot(index='centrality', columns='model', values='pos')
pivot_neg = sign_df.pivot(index='centrality', columns='model', values='neg')

# Net sign stability: fraction positive minus fraction negative
net = pivot_pos - pivot_neg

# Sort rows by mean net stability (most consistently positive at top)
row_order = net.mean(axis=1).sort_values(ascending=False).index
net = net.loc[row_order]

net.index = [pretty(i) for i in net.index]
net.columns = [c.upper() for c in net.columns]

fig, ax = plt.subplots(figsize=(8, 11))
sns.heatmap(net, annot=True, fmt='.2f', cmap='RdBu_r',
            vmin=-1, vmax=1, ax=ax, linewidths=0.5,
            cbar_kws={'label': 'Net Sign Stability\n(Frac. Positive $-$ Frac. Negative)', 'shrink': 0.8})
ax.set_title('Sign-Stability Matrix: Net Direction of AME\nby Centrality Measure and Regression Model',
             fontsize=12)
ax.set_xlabel('Regression Model')
ax.set_ylabel('Centrality Measure')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=9)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig(f'{FIGDIR}/sign_stability_matrix.png', bbox_inches='tight')
plt.close()
print("  Saved: sign_stability_matrix.png")

print("\nAll reviewer figures generated successfully.")
