#!/usr/bin/env python3
"""
Regenerate all figures for the Karnataka Multiverse Analysis.
Fixes the extreme outlier scale issue in ame_by_centrality and
ame_specification_curve by winsorising at the 99th percentile.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os

FIGDIR  = '/home/ubuntu/karnataka_multiverse/results/python/figs'
TABDIR  = '/home/ubuntu/karnataka_multiverse/results/python/tables'
os.makedirs(FIGDIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 150,
})

# ── Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv(f'{TABDIR}/regression_multiverse.csv')
print(f"Loaded {len(df)} specifications.")
print("Columns:", df.columns.tolist())

# Identify and report the outlier
abs_ame = df['ame'].abs()
p99 = abs_ame.quantile(0.999)
outliers = df[abs_ame > p99]
print(f"\nOutliers (|AME| > 99.9th pct = {p99:.4g}):")
print(outliers[['representation','centrality','outcome','model','ame']].to_string())

# Winsorise for plotting (keep original for tables)
df['ame_plot'] = df['ame'].clip(lower=-p99, upper=p99)
df_clean = df[abs_ame <= p99].copy()
print(f"\nClean specs after removing outliers: {len(df_clean)}")

# Centrality family mapping
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
    'coreness':             'Degree/Local',
}
df_clean['family'] = df_clean['centrality'].map(FAMILY).fillna('Other')
df['family'] = df['centrality'].map(FAMILY).fillna('Other')

FAMILY_COLORS = {
    'Degree/Local':      '#2166ac',
    'Distance':          '#d6604d',
    'Spectral/Prestige': '#4dac26',
    'Structural Holes':  '#8073ac',
    'Other':             '#888888',
}

# ── Figure 1: Outcome distributions ────────────────────────────────────────
# (already looks fine, just re-save at consistent DPI)
# We regenerate from the graph data
import pickle, networkx as nx
with open('/home/ubuntu/karnataka_multiverse/data/loan_nomination_graph.pkl','rb') as f:
    G = pickle.load(f)

nodes = list(G.nodes())
in_deg = dict(G.in_degree(weight='weight'))

# Village membership
village_of = lambda n: n.split('_')[0]
villages = list(set(village_of(n) for n in nodes))

raw_count = np.array([in_deg.get(n, 0) for n in nodes])

# Binary
binary = (raw_count > 0).astype(float)

# Within-village share and rank
village_share = np.zeros(len(nodes))
village_rank  = np.zeros(len(nodes))
for v in villages:
    vnodes = [n for n in nodes if village_of(n) == v]
    idx    = [nodes.index(n) for n in vnodes]
    vals   = raw_count[idx]
    total  = vals.sum()
    if total > 0:
        for i, ix in enumerate(idx):
            village_share[ix] = vals[i] / total
    ranks = pd.Series(vals).rank(pct=True).values
    for i, ix in enumerate(idx):
        village_rank[ix] = ranks[i]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Distribution of Outcome Variables — Karnataka Loan Nominations', fontsize=14)

outcomes = [
    (raw_count,     'Raw Nomination Count',        'steelblue'),
    (binary,        'Binary (Nominated at All)',    'steelblue'),
    (village_share, 'Within-Village Share',         'steelblue'),
    (village_rank,  'Within-Village Percentile Rank','steelblue'),
]
for ax, (vals, title, color) in zip(axes.flat, outcomes):
    pct_nonzero = (vals > 0).mean() * 100
    ax.hist(vals, bins=40, color=color, edgecolor='white', linewidth=0.3)
    ax.set_title(title)
    ax.set_xlabel('Value')
    ax.set_ylabel('Count')
    ax.text(0.97, 0.95, f'{pct_nonzero:.1f}% > 0',
            transform=ax.transAxes, ha='right', va='top',
            color='crimson', fontsize=9)

plt.tight_layout()
plt.savefig(f'{FIGDIR}/outcome_distributions.png', bbox_inches='tight')
plt.close()
print("Saved: outcome_distributions.png")

# ── Figure 2: Edge count across representations ─────────────────────────────
rep_edges = {
    'directed\_weighted':   G.number_of_edges(),
    'directed\_binary':     G.number_of_edges(),
    'undirected\_weighted': nx.Graph(G).number_of_edges(),
    'undirected\_binary':   nx.Graph(G).number_of_edges(),
    'directed\_strong\_ties': sum(1 for u,v,d in G.edges(data=True) if d.get('weight',1) >= np.percentile([d.get('weight',1) for _,_,d in G.edges(data=True)], 75)),
}
fig, ax = plt.subplots(figsize=(9, 4))
labels = list(rep_edges.keys())
vals   = list(rep_edges.values())
colors = ['#2166ac','#4393c3','#92c5de','#d1e5f0','#f4a582']
bars = ax.barh(labels, vals, color=colors, edgecolor='white')
ax.set_xlabel('Number of Edges')
ax.set_title('Edge Count Across Network Representations')
for bar, val in zip(bars, vals):
    ax.text(val + 200, bar.get_y() + bar.get_height()/2,
            f'{val:,}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig(f'{FIGDIR}/universe_edges_hist.png', bbox_inches='tight')
plt.close()
print("Saved: universe_edges_hist.png")

# ── Figure 3: AME by centrality (FIXED — winsorised, coloured by family) ───
# Order centralities by median AME (clean data)
cent_order = (df_clean.groupby('centrality')['ame']
              .median()
              .sort_values()
              .index.tolist())

fig, ax = plt.subplots(figsize=(16, 7))
for i, cent in enumerate(cent_order):
    sub = df_clean[df_clean['centrality'] == cent]['ame'].values
    fam = df_clean[df_clean['centrality'] == cent]['family'].iloc[0]
    color = FAMILY_COLORS.get(fam, '#888888')
    bp = ax.boxplot(sub, positions=[i], widths=0.6, patch_artist=True,
                    medianprops=dict(color='black', linewidth=1.5),
                    boxprops=dict(facecolor=color, alpha=0.7),
                    whiskerprops=dict(color=color),
                    capprops=dict(color=color),
                    flierprops=dict(marker='o', markersize=2,
                                   markerfacecolor=color, alpha=0.4))

ax.axhline(0, color='red', linestyle='--', linewidth=0.8, alpha=0.7)
ax.set_xticks(range(len(cent_order)))
ax.set_xticklabels([c.replace('_',' ') for c in cent_order],
                   rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Average Marginal Effect (AME)')
ax.set_title('Distribution of AMEs by Centrality Measure\n'
             '(across outcome variants and regression models; '
             'extreme outliers winsorised at 99.9th percentile)',
             fontsize=12)

# Legend for families
legend_patches = [mpatches.Patch(color=c, label=f, alpha=0.7)
                  for f, c in FAMILY_COLORS.items() if f != 'Other']
ax.legend(handles=legend_patches, loc='upper left', fontsize=8,
          title='Centrality Family', title_fontsize=8)

# Annotate n outliers removed
n_out = len(outliers)
if n_out > 0:
    ax.text(0.99, 0.01,
            f'{n_out} extreme outlier(s) excluded from display\n'
            f'(|AME| > {p99:.2e})',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=7, color='grey', style='italic')

plt.tight_layout()
plt.savefig(f'{FIGDIR}/ame_by_centrality.png', bbox_inches='tight')
plt.close()
print("Saved: ame_by_centrality.png (FIXED)")

# ── Figure 4: Specification curve (FIXED — winsorised) ──────────────────────
df_sorted = df_clean.sort_values('ame').reset_index(drop=True)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10),
                                gridspec_kw={'height_ratios': [3, 2]})
fig.suptitle('Specification Curve: Effect of Network Centrality on Loan Nominations\n'
             '(extreme outliers excluded; blue = p<0.05, grey = not significant)',
             fontsize=12)

colors_sig = ['#2166ac' if s else '#aaaaaa' for s in df_sorted['significant']]
ax1.scatter(range(len(df_sorted)), df_sorted['ame'],
            c=colors_sig, s=6, alpha=0.7, linewidths=0)
ax1.axhline(0, color='black', linestyle='--', linewidth=0.8)
ax1.set_ylabel('Average Marginal Effect (AME)')
ax1.set_xlim(-5, len(df_sorted)+5)
ax1.set_xticks([])

# Lower panel: model and outcome indicators
model_y   = {'ols': 0, 'poisson': 1, 'negbin': 2, 'logit': 3}
outcome_y = {'raw_count': 4, 'binary': 5, 'village_share': 6, 'village_pctrank': 7}

for idx, row in df_sorted.iterrows():
    m_y = model_y.get(row['model'], None)
    o_y = outcome_y.get(row['outcome'], None)
    col = '#2166ac' if row['significant'] else '#f4a582'
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
print("Saved: ame_specification_curve.png (FIXED)")

# ── Figure 5: Centrality stability heatmap ──────────────────────────────────
# Already looks good — re-save at consistent settings
stab = pd.read_csv(f'{TABDIR}/centrality_stability_top20.csv', index_col=0)
fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(stab.astype(float), annot=True, fmt='.2f', cmap='YlOrRd',
            vmin=0, vmax=1, ax=ax, linewidths=0.5,
            cbar_kws={'label': 'Jaccard overlap of Top-20 PageRank nodes'})
ax.set_title('Centrality Stability Heatmap (PageRank, Top-20)', fontsize=13)
ax.set_xticklabels([l.get_text().replace('_',' ') for l in ax.get_xticklabels()],
                   rotation=45, ha='right', fontsize=9)
ax.set_yticklabels([l.get_text().replace('_',' ') for l in ax.get_yticklabels()],
                   rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig(f'{FIGDIR}/centrality_stability_heatmap.png', bbox_inches='tight')
plt.close()
print("Saved: centrality_stability_heatmap.png")

print("\nAll figures regenerated successfully.")
