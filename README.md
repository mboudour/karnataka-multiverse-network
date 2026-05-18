# Multiverse Analysis of the Karnataka Loan Nomination Network

**Authors:** Moses Boudourides and Cristobal Young  
**Status:** Workflow Draft — May 2026

---

## Overview

This repository contains all code, data, and results for a comprehensive multiverse analysis of the **Karnataka Loan Nomination Network**, originally collected by Banerjee, Chandrasekhar, Duflo, and Jackson (2013) to study the diffusion of microfinance across 33 rural villages in Karnataka, India.

Following the multiverse analysis paradigm of Young and Cumberworth (2025) and Steegen et al. (2016), we systematically vary four dimensions of analytical choice — network representation, centrality measure, outcome variable, and regression model — to assess which conclusions about the structural determinants of loan nominations are robust and which are artifacts of specific methodological decisions.

The full pipeline produces **780 distinct regression specifications** and a **multiverse of Exponential Random Graph Models (ERGMs)** fitted village-by-village across the 33 disconnected village components.

> **Note:** The `.tex` and `.bib` paper source files are **not tracked** in this repository. The compiled PDF will be deposited in `paper/` after local compilation.

---

## Repository Structure

```
karnataka-multiverse-network/
│
├── data/
│   ├── loan_nomination_graph.pkl          # Original network (NetworkX DiGraph, pickled)
│   └── loan_nomination_edgelist.csv       # Edge list exported for R (from, to, weight)
│
├── python/
│   ├── multiverse_pipeline.py             # Main Python multiverse pipeline (780 specs)
│   ├── export_edgelist.py                 # Helper: exports .pkl to CSV for R
│   ├── regenerate_figures.py              # Regenerate all main figures
│   ├── fix_ame_figures.py                 # Fix AME figures (outlier removal)
│   ├── regenerate_tables.py               # Regenerate clean AME summary tables
│   └── generate_reviewer_figures.py       # Centrality correlation heatmap + sign-stability matrix
│
├── r/
│   ├── ergm_multiverse.R                  # CANONICAL: MCMLE, m1-m3 all villages, m4 top-10 only
│   ├── ergm_full.R                        # Complete MCMLE (all specs, all villages; very slow)
│   └── ergm_mple.R                        # Fast MPLE approximation (all specs, all villages)
│
├── results/
│   ├── python/
│   │   ├── tables/
│   │   │   ├── regression_multiverse.csv          # All 780 AME estimates
│   │   │   ├── ame_summary.csv / .tex             # Top specifications by AME (clean)
│   │   │   ├── centrality_universe_stats.csv      # Centrality stats per representation
│   │   │   ├── centrality_stability_top20.csv     # Node stability across representations
│   │   │   ├── centrality_ame_spearman_corr.csv   # Spearman AME correlation matrix
│   │   │   ├── pagerank_stability.tex             # PageRank stability table (LaTeX)
│   │   │   └── universe_summary.tex               # Network representation summary (LaTeX)
│   │   └── figs/
│   │       ├── ame_specification_curve.png              # Specification curve (all 780 specs)
│   │       ├── ame_by_centrality.png                    # AME distribution by centrality family
│   │       ├── centrality_stability_heatmap.png         # Heatmap of top-node stability
│   │       ├── outcome_distributions.png                # Distribution of the 4 outcome variants
│   │       ├── universe_edges_hist.png                  # Edge count across representations
│   │       ├── centrality_correlation_heatmap.png       # Spearman AME correlation among centralities
│   │       └── sign_stability_matrix.png                # Net sign direction of AME by centrality × model
│   └── r/
│       └── tables/
│           ├── ergm_multiverse_coefficients.csv   # Village-level ERGM coefficients
│           ├── ergm_pooled_estimates.csv          # Meta-analytic pooled estimates
│           └── ergm_fit_summary.csv               # Mean AIC/BIC by spec (when complete)
│
├── paper/
│   └── karnataka_multiverse_analysis.pdf  # Compiled paper PDF (to be uploaded after local compilation)
│
└── README.md
```

---

## Network Description

The Karnataka Loan Nomination Network is a **directed, weighted graph** in which nodes are households and a directed edge from household $i$ to household $j$ indicates that $i$ nominated $j$ as a reliable source of loan advice. The network has the following properties:

| Property | Value |
|---|---|
| Nodes (households) | 7,311 |
| Edges (nominations) | 17,538 |
| Villages (components) | 33 |
| Overall density | 0.0003 |
| Households nominated (≥1) | ~4% |

The 33 villages are **completely disconnected** from one another — all ties are strictly within-village. This structural feature is central to all analytical decisions in this project.

---

## Multiverse Dimensions

### 1. Network Representation (5 variants)
| Label | Description |
|---|---|
| Directed Weighted | Raw network with original edge weights |
| Directed Binary | Weights binarised (0/1) |
| Undirected Weighted | Symmetrised by summing weights in both directions |
| Undirected Binary | Tie exists if nomination occurred in either direction |
| Directed Strong Ties | Only top-quartile weight ties retained ($w \geq 0.75$) |

### 2. Centrality Measures (26 measures)
Computed for each representation across four theoretical families:

- **Degree/Local:** In-Degree, Out-Degree, Strength, Total Degree, k-Core, Onion Layer
- **Distance/Geodesic:** Harmonic Centrality, Closeness Centrality, Betweenness (sampled, $k=500$), Load Centrality
- **Spectral/Prestige:** PageRank, Eigenvector Centrality, Katz Centrality, HITS Hubs, HITS Authorities, Proximity Prestige
- **Structural Holes:** Effective Size, Constraint (Burt 2004)
- **Additional:** Triangles, Square Clustering, Average Neighbor Degree, Weighted Degree, Clustering Coefficient, Coreness, Eccentricity (approximated)

### 3. Outcome Variable (4 variants)
| Label | Description |
|---|---|
| Raw Count | Integer nomination count received |
| Binary | 1 if nominated at all, 0 otherwise |
| Within-Village Rank | Percentile rank within village |
| Within-Village Share | Fraction of village nominations received |

### 4. Regression Model (4 models, all with village fixed effects)
| Model | Outcome | Notes |
|---|---|---|
| OLS | Count, Rank, Share | Average Marginal Effect = coefficient |
| Poisson | Count | AME computed via `statsmodels` |
| Negative Binomial | Count | AME computed via `statsmodels` |
| Logistic | Binary | AME computed via `statsmodels` |

All non-linear models report **Average Marginal Effects (AMEs)** to ensure comparability across specifications.

---

## ERGM Multiverse

ERGMs are fitted independently for each village and pooled via **inverse-variance meta-analysis**. Two network representations (directed, undirected) are crossed with nested specifications:

| Spec | Directed Terms | Undirected Terms | Villages |
|---|---|---|---|
| m1 | `edges` | `edges` | All 33 |
| m2 | `edges + mutual` | `edges + gwdegree` | All 33 |
| m3 | `edges + mutual + gwodegree` | `edges + gwesp`$^\dagger$ | All 33 / Top-10$^\dagger$ |
| m4 | `edges + mutual + gwesp`$^\dagger$ | — | Top-10$^\dagger$ |

$^\dagger$ **Specifications involving `gwesp`** (geometrically-weighted edgewise shared partners) require full MCMC sampling over a triangle-counting sufficient statistic and are computationally prohibitive at the scale of all 33 villages. These specifications were restricted to the **10 largest villages by household count**: v52 (375), v65 (362), v59 (353), v71 (310), v43 (308), v50 (298), v55 (271), v45 (269), v76 (267), v40 (262). During estimation, the `gwesp` term exhibited **severe linear dependence** with preceding statistics, indicating non-identifiability due to extreme triangle sparsity in these village networks. The `gwesp` specifications are therefore omitted from the final pooled results and reported as non-identifiable in the paper.

All ERGM estimation uses MCMLE with up to 20 iterations. Non-`gwesp` specs use MCMC sample size 1,024; `gwesp` specs use sample size 2,048 with burn-in 10,000.

### ERGM R Scripts

Three R scripts are provided:

| Script | Method | Specs | Villages | Use case |
|---|---|---|---|---|
| `ergm_multiverse.R` | MCMLE | m1–m3 directed, m1–m2 undirected | All 33 | **Recommended** — canonical analysis |
| `ergm_full.R` | MCMLE | m1–m4 directed, m1–m3 undirected | All 33 | Complete but very slow (24–72h) |
| `ergm_mple.R` | MPLE | m1–m4 directed, m1–m3 undirected | All 33 | Fast screening (minutes); SEs not valid for inference |

---

## Reproducing the Analysis

### Requirements

**Python (≥ 3.9):**
```
networkx, pandas, numpy, scipy, statsmodels, matplotlib, seaborn
```

**R (≥ 4.1):**
```
ergm (3.11.x), network, statnet
```

### Steps

```bash
# 1. Export edge list from pickle (required for R)
python3 python/export_edgelist.py

# 2. Run the full Python multiverse pipeline
python3 python/multiverse_pipeline.py
# Outputs: results/python/tables/ and results/python/figs/

# 3. Generate reviewer figures (correlation heatmap, sign-stability matrix)
python3 python/generate_reviewer_figures.py

# 4. Run the ERGM multiverse
Rscript r/ergm_multiverse.R data/loan_nomination_edgelist.csv results/r
# Outputs: results/r/tables/
```

---

## Key Findings (Workflow Draft)

- **Effective Size** (structural holes) and **In-Degree** are the most robust predictors of receiving loan nominations, appearing at the top of the specification curve across the majority of the 780 specifications.
- **Betweenness Centrality** yields highly unstable and often insignificant effects — consistent with the fragmented, village-bounded structure of the network, where global bridging is structurally impossible.
- **Model choice** (OLS vs. Poisson vs. Negative Binomial) is the single largest driver of AME scale. The Negative Binomial model diverged pathologically for 28 specifications under extreme zero-inflation, producing unbounded AMEs — a substantive finding about the limits of standard count models for sparse nomination data.
- **ERGM results** reveal a sparse, non-reciprocal network: the negative `mutual` coefficient indicates that households are *less* likely to mutually nominate each other than chance would predict, suggesting a hierarchical flow of loan advice rather than reciprocal exchange. The `gwesp` term was found to be non-identifiable due to extreme triangle sparsity.
- **Node rankings are fragile:** only household `v56_h66` appears in the top-20 by PageRank across all five network representations.
- **Centrality collinearity:** The centrality correlation heatmap reveals strong clustering among degree-family measures and among prestige/spectral measures, while Constraint and Clustering form a distinct anti-correlated cluster.

---

## Citation

If you use this code or results, please cite:

> Boudourides, Moses and Young, Cristobal (2026). *A Multiverse Analysis of the Karnataka Loan Nomination Network*. Workflow Draft.

And the original data source:

> Banerjee, A., Chandrasekhar, A. G., Duflo, E., and Jackson, M. O. (2013). The Diffusion of Microfinance. *Science*, 341(6144), 1236498.

---

## License

Code: MIT License. Data: subject to the original data use terms of Banerjee et al. (2013).
