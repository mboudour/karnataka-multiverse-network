#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os

TABDIR = '/home/ubuntu/karnataka_multiverse/results/python/tables'

# Load the full multiverse
df = pd.read_csv(f'{TABDIR}/regression_multiverse.csv')

# Remove the broken NegBin specs (overflow/inf)
df_clean = df[np.isfinite(df['ame']) & (df['ame'].abs() < 1e6)].copy()

# Group by centrality, outcome, and model to get mean AME and fraction significant
summary = (df_clean.groupby(['centrality', 'outcome', 'model'])
           .agg(
               mean_ame=('ame', 'mean'),
               frac_sig=('significant', 'mean'),
               n_specs=('ame', 'count')
           )
           .reset_index())

# Sort by mean AME descending to get the top positive predictors
summary = summary.sort_values('mean_ame', ascending=False).reset_index(drop=True)

# Format the table for LaTeX
top30 = summary.head(30).copy()
top30['mean_ame'] = top30['mean_ame'].apply(lambda x: f"{x:.4f}")
top30['frac_sig'] = top30['frac_sig'].apply(lambda x: f"{x:.4f}")

# Rename columns for LaTeX
top30 = top30.rename(columns={
    'centrality': 'Centrality',
    'outcome': 'Outcome',
    'model': 'Model',
    'mean_ame': 'Mean AME',
    'frac_sig': 'Frac.\\textbackslash space Sig.',
    'n_specs': 'N'
})

# Escape underscores
top30['Centrality'] = top30['Centrality'].str.replace('_', '\\_')
top30['Outcome'] = top30['Outcome'].str.replace('_', '\\_')

# Write to tex
latex_str = "\\begin{table}[htbp]\n\\centering\n"
latex_str += "\\caption{Top 30 universe-level AME summaries (sorted by mean AME; excludes 28 divergent Negative Binomial specifications).}\n"
latex_str += "\\label{tab:ame_summary}\n"
latex_str += top30.to_latex(index=False, column_format="lllrrc", escape=False)
latex_str += "\\end{table}\n"

with open(f'{TABDIR}/ame_summary.tex', 'w') as f:
    f.write(latex_str)

print("Regenerated ame_summary.tex with clean data.")
