#!/usr/bin/env python3
"""Fix ame_by_centrality and ame_specification_curve: remove inf/nan and overflow outliers."""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

FIGDIR = '/home/ubuntu/karnataka_multiverse/results/python/figs'
TABDIR = '/home/ubuntu/karnataka_multiverse/results/python/tables'

df = pd.read_csv(f'{TABDIR}/regression_multiverse.csv')

# Step 1: remove inf, nan, and numerical overflow (|AME| > 1e6 is certainly unphysical)
df_valid = df[np.isfinite(df['ame']) & (df['ame'].abs() < 1e6)].copy()
n_removed = len(df) - len(df_valid)
print(f"Total specs: {len(df)}, valid after removing overflow/inf: {len(df_valid)}")
print(f"Removed {n_removed} overflow/inf specs:")
removed = df[~(np.isfinite(df['ame']) & (df['ame'].abs() < 1e6))]
print(removed[['representation','centrality','outcome','model','ame']].to_string())

# Step 2: further winsorise at 99th percentile of the valid set for display
p99 = df_valid['ame'].abs().quantile(0.95)  # 95th pct for cleaner display
df_plot = df_valid.copy()
df_plot['ame_plot'] = df_plot['ame'].clip(-p99, p99)
print(f"\n99th pct of valid AMEs: {p99:.6f}")
print(f"AME range for plotting: [{-p99:.6f}, {p99:.6f}]")

FAMILY = {
    'in_degree':            'Degree/Local',
    'out_degree':           'Degree/Local',
    'total_degree':         'Degree/Local',
    'strength':             'Degree/Local',
    'weighted_in_degree':   'Degree/Local',
    'weighted_out_degree':  'Degree/Local',
    'k_core':               'Degree/Local',
    'onion_layer':          'Degree/Local',
    'clustering':           'Degree/Local',
    'triangles':            'Degree/Local',
    'square_clustering':    'Degree/Local',
    'avg_neighbor_degree':  'Degree/Local',
    'coreness':             'Degree/Local',
    'harmonic_centrality':  'Distance',
    'closeness_centrality': 'Distance',
    'betweenness_k500':     'Distance',
    'load_centrality':      'Distance',
    'pagerank':             'Spectral/Prestige',
    'pagerank_reverse':     'Spectral/Prestige',
    'eigenvector':          'Spectral/Prestige',
    'katz_centrality':      'Spectral/Prestige',
    'katz_prestige_in':     'Spectral/Prestige',
    'hits_hub':             'Spectral/Prestige',
    'hits_authority':       'Spectral/Prestige',
    'proximity_prestige':   'Spectral/Prestige',
    'effective_size':       'Structural Holes',
    'constraint':           'Structural Holes',
}
FAMILY_COLORS = {
    'Degree/Local':      '#2166ac',
    'Distance':          '#d6604d',
    'Spectral/Prestige': '#4dac26',
    'Structural Holes':  '#8073ac',
    'Other':             '#888888',
}
df_plot['family'] = df_plot['centrality'].map(FAMILY).fillna('Other')

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 150,
})

# ── Figure: AME by centrality (boxplot, coloured by family) ─────────────────
cent_order = (df_plot.groupby('centrality')['ame_plot']
              .median().sort_values().index.tolist())

fig, ax = plt.subplots(figsize=(16, 7))
for i, cent in enumerate(cent_order):
    sub = df_plot[df_plot['centrality'] == cent]['ame_plot'].values
    fam = df_plot[df_plot['centrality'] == cent]['family'].iloc[0]
    color = FAMILY_COLORS.get(fam, '#888888')
    ax.boxplot(sub, positions=[i], widths=0.6, patch_artist=True,
               medianprops=dict(color='black', linewidth=1.5),
               boxprops=dict(facecolor=color, alpha=0.7),
               whiskerprops=dict(color=color),
               capprops=dict(color=color),
               flierprops=dict(marker='o', markersize=2,
                               markerfacecolor=color, alpha=0.4))

ax.axhline(0, color='red', linestyle='--', linewidth=0.8, alpha=0.7)
ax.set_xticks(range(len(cent_order)))
ax.set_xticklabels([c.replace('_', ' ') for c in cent_order],
                   rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Average Marginal Effect (AME)')
ax.set_title(
    'Distribution of AMEs by Centrality Measure\n'
    '(across outcome variants and regression models; '
    f'{n_removed} overflow spec(s) excluded; display winsorised at 95th pct)',
    fontsize=11)

legend_patches = [mpatches.Patch(color=c, label=f, alpha=0.7)
                  for f, c in FAMILY_COLORS.items() if f != 'Other']
ax.legend(handles=legend_patches, loc='upper left', fontsize=8,
          title='Centrality Family', title_fontsize=8)

plt.tight_layout()
plt.savefig(f'{FIGDIR}/ame_by_centrality.png', bbox_inches='tight')
plt.close()
print("Saved: ame_by_centrality.png")

# ── Figure: Specification curve ──────────────────────────────────────────────
df_sorted = df_plot.sort_values('ame_plot').reset_index(drop=True)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10),
                                gridspec_kw={'height_ratios': [3, 2]})
fig.suptitle(
    'Specification Curve: Effect of Network Centrality on Loan Nominations\n'
    f'({n_removed} NegBin overflow specs excluded; display winsorised at 95th pct; blue = p<0.05, grey = not significant)',
    fontsize=11)

colors_sig = ['#2166ac' if s else '#aaaaaa' for s in df_sorted['significant']]
ax1.scatter(range(len(df_sorted)), df_sorted['ame_plot'],
            c=colors_sig, s=6, alpha=0.7, linewidths=0)
ax1.axhline(0, color='black', linestyle='--', linewidth=0.8)
ax1.set_ylabel('Average Marginal Effect (AME)')
ax1.set_xlim(-5, len(df_sorted)+5)
ax1.set_xticks([])

model_y   = {'ols': 0, 'poisson': 1, 'negbin': 2, 'logit': 3}
outcome_y = {'raw_count': 4, 'binary': 5, 'village_share': 6, 'village_pctrank': 7}

for idx, row in df_sorted.iterrows():
    col = '#2166ac' if row['significant'] else '#f4a582'
    m_y = model_y.get(row['model'])
    o_y = outcome_y.get(row['outcome'])
    if m_y is not None:
        ax2.scatter(idx, m_y, c=col, s=4, alpha=0.6, linewidths=0)
    if o_y is not None:
        ax2.scatter(idx, o_y, c=col, s=4, alpha=0.6, linewidths=0)

all_y_labels = list(model_y.keys()) + list(outcome_y.keys())
all_y_pos    = list(model_y.values()) + list(outcome_y.values())
ax2.set_yticks(all_y_pos)
ax2.set_yticklabels(all_y_labels, fontsize=8)
ax2.set_ylabel('Model / Outcome')
ax2.set_xlim(-5, len(df_sorted)+5)
ax2.set_xlabel('Specification (sorted by AME)')

plt.tight_layout()
plt.savefig(f'{FIGDIR}/ame_specification_curve.png', bbox_inches='tight')
plt.close()
print("Saved: ame_specification_curve.png")
print("Done.")
