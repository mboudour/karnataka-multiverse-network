#!/usr/bin/env python3
"""
Multiverse Analysis of the Karnataka Loan Nomination Network
============================================================
Augmented pipeline: computes a rich set of centrality measures across multiple
network representations, constructs outcome variants (count, binary, within-village
share, within-village percentile rank), and runs a regression multiverse
(OLS, Poisson, Negative Binomial, Logit) with average marginal effects (AMEs)
so that coefficients are comparable across models.

Usage:
    python3 python/multiverse_pipeline.py \
        --graph data/loan_nomination_graph.pkl \
        --outdir results/python

Outputs (in outdir):
    tables/centrality_universe_stats.csv
    tables/centrality_stability_top20.csv
    tables/regression_multiverse.csv
    tables/ame_summary.csv
    tables/universe_summary.tex
    tables/pagerank_stability.tex
    tables/ame_summary.tex
    figs/centrality_stability_heatmap.png
    figs/universe_edges_hist.png
    figs/ame_specification_curve.png
    figs/ame_by_centrality.png
    figs/outcome_distributions.png
"""
from __future__ import annotations
import argparse, pickle, re, warnings, sys
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. CENTRALITY MEASURES
# ─────────────────────────────────────────────

def compute_centralities(G: nx.DiGraph, slow: bool = True) -> pd.DataFrame:
    """
    Compute all feasible centrality measures on a directed weighted graph.
    slow=False skips percolation_centrality (takes ~35s on this graph).
    Returns a DataFrame with one row per node.
    """
    G_und = G.to_undirected()
    rows = {n: {} for n in G.nodes()}

    def store(name, vals):
        if vals is None:
            return
        for n, v in vals.items():
            if n in rows:
                rows[n][name] = float(v) if v is not None else np.nan

    # --- Degree family ---
    store("in_degree",            dict(G.in_degree()))
    store("out_degree",           dict(G.out_degree()))
    store("total_degree",         dict(G.degree()))
    store("weighted_in_degree",   dict(G.in_degree(weight="weight")))
    store("weighted_out_degree",  dict(G.out_degree(weight="weight")))
    store("strength",             dict(G.degree(weight="weight")))
    store("avg_neighbor_degree",  nx.average_neighbor_degree(G, weight="weight"))
    store("k_core",               nx.core_number(G_und))
    store("onion_layer",          nx.onion_layers(G_und))

    # --- Distance / geodesic family ---
    store("harmonic_centrality",  nx.harmonic_centrality(G))
    store("closeness_centrality", nx.closeness_centrality(G))
    store("betweenness_k500",     nx.betweenness_centrality(
        G, k=500, weight="weight", normalized=True, seed=42))
    store("load_centrality",      nx.load_centrality(G, weight="weight"))

    # --- Spectral / eigenvector family ---
    store("pagerank",             nx.pagerank(G, weight="weight"))
    store("pagerank_reverse",     nx.pagerank(G.reverse(), weight="weight"))
    try:
        h, a = nx.hits(G, max_iter=1000, tol=1e-6, normalized=True)
        store("hits_hub",         h)
        store("hits_authority",   a)
    except Exception:
        pass
    try:
        store("eigenvector",      nx.eigenvector_centrality(
            G, weight="weight", max_iter=1000))
    except Exception:
        pass
    try:
        from scipy.sparse.linalg import eigs as sp_eigs
        A = nx.to_scipy_sparse_array(G, weight="weight", format="csr")
        vals, _ = sp_eigs(A, k=1, which="LM", return_eigenvectors=True)
        lmax = abs(vals[0])
        alpha_k = 0.85 / lmax if lmax > 0 else 0.01
        store("katz_centrality",  nx.katz_centrality(
            G, alpha=alpha_k, weight="weight", max_iter=1000, tol=1e-4))
        store("katz_prestige_in", nx.katz_centrality(
            G.reverse(), alpha=alpha_k, weight="weight", max_iter=1000, tol=1e-4))
    except Exception:
        pass

    # --- Structural holes (Burt) ---
    store("constraint",           nx.constraint(G, weight="weight"))
    store("effective_size",       nx.effective_size(G, weight="weight"))

    # --- Local structure ---
    store("clustering",           nx.clustering(G_und, weight="weight"))
    store("triangles",            nx.triangles(G_und))
    store("square_clustering",    nx.square_clustering(G_und))

    # --- Prestige ---
    store("proximity_prestige",   _proximity_prestige(G))

    # --- Percolation (slow: ~35s on full graph) ---
    if slow:
        try:
            store("percolation",  nx.percolation_centrality(G, attribute=None))
        except Exception:
            pass

    return pd.DataFrame.from_dict(rows, orient="index")


def _proximity_prestige(G: nx.DiGraph) -> dict:
    n = G.number_of_nodes()
    pp = {}
    for node in G.nodes():
        lengths = dict(nx.single_target_shortest_path_length(G, node))
        reachable = {k: v for k, v in lengths.items() if k != node and v > 0}
        if not reachable:
            pp[node] = 0.0
        else:
            frac = len(reachable) / (n - 1)
            avg_d = np.mean(list(reachable.values()))
            pp[node] = frac / avg_d if avg_d > 0 else 0.0
    return pp


# ─────────────────────────────────────────────
# 2. NETWORK REPRESENTATIONS
# ─────────────────────────────────────────────

def make_representations(G: nx.DiGraph) -> dict[str, nx.DiGraph]:
    """Return a dict of named network representations."""
    reps = {}

    # Weighted directed (original)
    reps["directed_weighted"] = G.copy()

    # Binary directed
    Gbin = G.copy()
    for u, v in Gbin.edges():
        Gbin[u][v]["weight"] = 1.0
    reps["directed_binary"] = Gbin

    # Undirected weighted
    G_und = G.to_undirected()
    reps["undirected_weighted"] = nx.DiGraph(G_und)

    # Undirected binary
    G_und_bin = G_und.copy()
    for u, v in G_und_bin.edges():
        G_und_bin[u][v]["weight"] = 1.0
    reps["undirected_binary"] = nx.DiGraph(G_und_bin)

    # Strong-tie threshold (top quartile of weights)
    weights = [d["weight"] for _, _, d in G.edges(data=True)]
    q75 = np.percentile(weights, 75)
    Gstrong = nx.DiGraph()
    Gstrong.add_nodes_from(G.nodes())
    for u, v, d in G.edges(data=True):
        if d["weight"] >= q75:
            Gstrong.add_edge(u, v, weight=d["weight"])
    reps["directed_strong_ties"] = Gstrong

    return reps


# ─────────────────────────────────────────────
# 3. OUTCOME VARIABLES
# ─────────────────────────────────────────────

def build_outcomes(G: nx.DiGraph) -> pd.DataFrame:
    """
    Derive four outcome variants from in-degree (nomination count).
    """
    village_re = re.compile(r"(v\d+)")
    rows = []
    for node in G.nodes():
        m = village_re.match(str(node))
        village = m.group(1) if m else "unknown"
        count = G.in_degree(node)
        rows.append({"node": node, "village": village, "raw_count": count})

    df = pd.DataFrame(rows).set_index("node")
    df["binary"] = (df["raw_count"] > 0).astype(int)

    village_totals = df.groupby("village")["raw_count"].transform("sum")
    df["village_share"] = df["raw_count"] / village_totals.replace(0, np.nan)

    df["village_pctrank"] = df.groupby("village")["raw_count"].rank(pct=True)

    return df


# ─────────────────────────────────────────────
# 4. REGRESSION MULTIVERSE WITH AMEs
# ─────────────────────────────────────────────

def run_regression_multiverse(
    centrality_df: pd.DataFrame,
    outcome_df: pd.DataFrame,
    rep_name: str,
) -> pd.DataFrame:
    """
    For each (centrality measure) × (outcome variant) × (regression model),
    fit the model and compute the Average Marginal Effect (AME).
    """
    import statsmodels.api as sm

    outcome_variants = ["raw_count", "binary", "village_share", "village_pctrank"]
    # model → applicable outcomes
    model_outcomes = {
        "ols":     ["raw_count", "village_share", "village_pctrank"],
        "poisson": ["raw_count"],
        "negbin":  ["raw_count"],
        "logit":   ["binary"],
    }

    merged = centrality_df.join(outcome_df[outcome_variants + ["village"]], how="inner")
    merged = merged.replace([np.inf, -np.inf], np.nan)

    records = []

    for cent in centrality_df.columns:
        if cent not in merged.columns:
            continue
        sub = merged[[cent] + outcome_variants + ["village"]].dropna(subset=[cent])
        if len(sub) < 50:
            continue

        x_raw = sub[cent].values.astype(float)
        x_std = (x_raw - x_raw.mean()) / (x_raw.std() + 1e-12)

        vill_dummies = pd.get_dummies(sub["village"], drop_first=True).values.astype(float)

        for model_name, applicable in model_outcomes.items():
            for outcome in applicable:
                y = sub[outcome].values.astype(float)
                valid = ~np.isnan(y)
                xv = x_std[valid]
                yv = y[valid]
                vd = vill_dummies[valid]
                if len(yv) < 50:
                    continue

                X = np.column_stack([np.ones(len(xv)), xv, vd])
                ame = se = pval = np.nan
                n_obs = int(valid.sum())

                try:
                    if model_name == "ols":
                        res = sm.OLS(yv, X).fit(cov_type="HC3")
                        ame  = res.params[1]
                        se   = res.bse[1]
                        pval = res.pvalues[1]

                    elif model_name == "poisson":
                        yc = np.round(yv).astype(int)
                        res = sm.GLM(yc, X, family=sm.families.Poisson()).fit(disp=False)
                        mu  = res.predict()
                        ame  = res.params[1] * mu.mean()
                        se   = res.bse[1]   * mu.mean()
                        pval = res.pvalues[1]

                    elif model_name == "negbin":
                        yc = np.round(yv).astype(int)
                        res = sm.NegativeBinomial(yc, X).fit(
                            disp=False, method="nm", maxiter=300)
                        mu  = res.predict()
                        ame  = res.params[1] * mu.mean()
                        se   = res.bse[1]   * mu.mean()
                        pval = res.pvalues[1]

                    elif model_name == "logit":
                        res = sm.Logit(yv, X).fit(disp=False, maxiter=300)
                        p   = res.predict()
                        ame  = res.params[1] * (p * (1 - p)).mean()
                        se   = res.bse[1]   * (p * (1 - p)).mean()
                        pval = res.pvalues[1]

                except Exception:
                    # OLS fallback
                    try:
                        coef, _, _, _ = np.linalg.lstsq(X, yv, rcond=None)
                        ame = coef[1]
                        resid = yv - X @ coef
                        s2 = resid.var()
                        xtx_inv = np.linalg.pinv(X.T @ X)
                        se = np.sqrt(s2 * xtx_inv[1, 1])
                        t = ame / (se + 1e-12)
                        pval = 2 * (1 - stats.t.cdf(abs(t), df=len(yv) - X.shape[1]))
                    except Exception:
                        pass

                records.append({
                    "representation": rep_name,
                    "centrality":     cent,
                    "outcome":        outcome,
                    "model":          model_name,
                    "ame":            ame,
                    "se":             se,
                    "pval":           pval,
                    "n_obs":          n_obs,
                    "significant":    bool(pval < 0.05) if not np.isnan(pval) else False,
                })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────
# 5. STABILITY ANALYSIS
# ─────────────────────────────────────────────

def centrality_stability(cent_dfs: dict[str, pd.DataFrame], k: int = 20) -> pd.DataFrame:
    all_measures = set()
    for df in cent_dfs.values():
        all_measures.update(df.columns)

    records = []
    for measure in sorted(all_measures):
        counter = Counter()
        total = 0
        for rep_name, df in cent_dfs.items():
            if measure not in df.columns:
                continue
            col = df[measure].dropna()
            if len(col) < k:
                continue
            for node in col.nlargest(k).index.tolist():
                counter[node] += 1
            total += 1
        for node, cnt in counter.most_common(50):
            records.append({
                "measure":     measure,
                "node":        node,
                "top_k_count": cnt,
                "top_k_freq":  cnt / total if total else np.nan,
                "n_universes": total,
            })
    return pd.DataFrame(records)


def jaccard_heatmap(cent_dfs, measure="pagerank", k=20):
    labels, tops = [], []
    for rep_name, df in cent_dfs.items():
        if measure not in df.columns:
            continue
        col = df[measure].dropna()
        if len(col) < k:
            continue
        labels.append(rep_name)
        tops.append(set(col.nlargest(k).index.tolist()))

    n = len(tops)
    jac = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            inter = len(tops[i] & tops[j])
            union = len(tops[i] | tops[j])
            jac[i, j] = inter / union if union else np.nan
    return labels, jac


# ─────────────────────────────────────────────
# 6. FIGURES
# ─────────────────────────────────────────────

def plot_stability_heatmap(labels, jac, measure, outpath):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(jac, aspect="auto", vmin=0, vmax=1, cmap="YlOrRd")
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"Jaccard overlap of Top-20 {measure} nodes", fontsize=9)
    ax.set_title(f"Centrality stability heatmap ({measure}, Top-20)", fontsize=11)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_universe_edges_hist(reps, outpath):
    names = list(reps.keys())
    edges = [reps[r].number_of_edges() for r in names]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(names, edges, color="steelblue")
    ax.set_xlabel("Number of edges")
    ax.set_title("Edge count across network representations")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_outcome_distributions(outcome_df, outpath):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    outcomes = ["raw_count", "binary", "village_share", "village_pctrank"]
    titles   = ["Raw Nomination Count", "Binary (Nominated at All)",
                 "Within-Village Share", "Within-Village Percentile Rank"]
    for ax, col, title in zip(axes.flat, outcomes, titles):
        data = outcome_df[col].dropna()
        ax.hist(data, bins=30, color="steelblue", edgecolor="white")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Value"); ax.set_ylabel("Count")
        pct_nz = (data > 0).mean() * 100
        ax.text(0.97, 0.95, f"{pct_nz:.1f}% > 0",
                transform=ax.transAxes, ha="right", va="top", fontsize=8, color="darkred")
    fig.suptitle("Distribution of Outcome Variables — Karnataka Loan Nominations",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_specification_curve(reg_df, outpath):
    df = reg_df.dropna(subset=["ame"]).copy()
    df = df.sort_values("ame").reset_index(drop=True)
    if df.empty:
        return

    fig = plt.figure(figsize=(14, 8))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[3, 2], hspace=0.05)
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1], sharex=ax_top)

    colors = df["significant"].map({True: "steelblue", False: "lightgray"})
    ax_top.scatter(df.index, df["ame"], c=colors, s=8, alpha=0.6, linewidths=0)
    ax_top.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax_top.set_ylabel("Average Marginal Effect (AME)", fontsize=10)
    ax_top.set_title(
        "Specification Curve: Effect of Network Centrality on Loan Nominations\n"
        "(blue = p<0.05; grey = not significant)", fontsize=11)

    model_map   = {"ols": 0, "poisson": 1, "negbin": 2, "logit": 3}
    outcome_map = {"raw_count": 0, "binary": 1, "village_share": 2, "village_pctrank": 3}
    for i, row in df.iterrows():
        c = "steelblue" if row["significant"] else "lightgray"
        ax_bot.scatter(i, model_map.get(row["model"], -1),
                       marker="|", color=c, s=12, linewidths=0.5)
        ax_bot.scatter(i, outcome_map.get(row["outcome"], -1) - 5,
                       marker="|", color="darkorange" if row["significant"] else "lightgray",
                       s=12, linewidths=0.5)

    yticks = list(model_map.values()) + [v - 5 for v in outcome_map.values()]
    ylabels = list(model_map.keys()) + list(outcome_map.keys())
    ax_bot.set_yticks(yticks)
    ax_bot.set_yticklabels(ylabels, fontsize=7)
    ax_bot.set_xlabel("Specification (sorted by AME)", fontsize=10)
    ax_bot.set_ylabel("Model / Outcome", fontsize=9)
    plt.setp(ax_top.get_xticklabels(), visible=False)

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_ame_by_centrality(reg_df, outpath):
    df = reg_df.dropna(subset=["ame"]).copy()
    if df.empty:
        return
    order = (df.groupby("centrality")["ame"]
               .median()
               .sort_values(ascending=False)
               .index.tolist())
    fig, ax = plt.subplots(figsize=(14, 6))
    data_by_cent = [df[df["centrality"] == c]["ame"].values for c in order]
    bp = ax.boxplot(data_by_cent, patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=1.5))
    for patch in bp["boxes"]:
        patch.set_facecolor("steelblue")
        patch.set_alpha(0.6)
    ax.axhline(0, color="red", linewidth=0.8, linestyle="--")
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(order, rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("Average Marginal Effect (AME)", fontsize=10)
    ax.set_title(
        "Distribution of AMEs by Centrality Measure\n"
        "(across outcome variants and regression models)", fontsize=11)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


# ─────────────────────────────────────────────
# 7. LATEX TABLE HELPERS
# ─────────────────────────────────────────────

def df_to_latex(df, caption, label, outpath, float_fmt="%.4f", col_format=None):
    n_cols = len(df.columns)
    if col_format is None:
        col_format = "l" + "r" * n_cols
    tex = df.to_latex(
        index=False,
        caption=caption,
        label=label,
        float_format=float_fmt.__mod__,
        column_format=col_format,
        escape=True,
        longtable=False,
    )
    Path(outpath).write_text(tex)


# ─────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph",  default="data/loan_nomination_graph.pkl")
    ap.add_argument("--outdir", default="results/python")
    ap.add_argument("--no-slow", action="store_true",
                    help="Skip slow centrality measures (percolation)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    (outdir / "tables").mkdir(parents=True, exist_ok=True)
    (outdir / "figs").mkdir(parents=True, exist_ok=True)

    # ── Load graph ──
    print("Loading graph …", flush=True)
    with open(args.graph, "rb") as f:
        G = pickle.load(f)
    print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}", flush=True)

    # ── Build outcome variables ──
    print("Building outcome variables …", flush=True)
    outcome_df = build_outcomes(G)
    zi = (outcome_df["raw_count"] == 0).mean() * 100
    print(f"  Zero-inflation (raw_count==0): {zi:.1f}%", flush=True)

    # ── Build representations ──
    print("Building network representations …", flush=True)
    reps = make_representations(G)
    print(f"  Representations: {list(reps.keys())}", flush=True)

    # ── Compute centralities per representation ──
    cent_dfs = {}
    for rep_name, Grep in reps.items():
        slow = not args.no_slow
        print(f"  Computing centralities for: {rep_name} …", flush=True)
        cent_dfs[rep_name] = compute_centralities(Grep, slow=slow)
        ncols = len(cent_dfs[rep_name].columns)
        print(f"    -> {ncols} measures computed", flush=True)

    # ── Universe stats table ──
    print("Building universe stats table …", flush=True)
    stats_rows = []
    for rep_name, df in cent_dfs.items():
        Grep = reps[rep_name]
        stats_rows.append({
            "Representation":      rep_name,
            "Nodes":               Grep.number_of_nodes(),
            "Edges":               Grep.number_of_edges(),
            "Density":             round(nx.density(Grep), 6),
            "Centrality Measures": len(df.columns),
        })
    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(outdir / "tables/centrality_universe_stats.csv", index=False)

    # ── Stability analysis ──
    print("Computing centrality stability …", flush=True)
    stab_df = centrality_stability(cent_dfs, k=20)
    stab_df.to_csv(outdir / "tables/centrality_stability_top20.csv", index=False)

    # ── Stability heatmap (PageRank) ──
    print("Plotting stability heatmap …", flush=True)
    labels, jac = jaccard_heatmap(cent_dfs, measure="pagerank", k=20)
    plot_stability_heatmap(labels, jac, "pagerank",
                           outdir / "figs/centrality_stability_heatmap.png")

    # ── Universe edges histogram ──
    plot_universe_edges_hist(reps, outdir / "figs/universe_edges_hist.png")

    # ── Outcome distributions ──
    print("Plotting outcome distributions …", flush=True)
    plot_outcome_distributions(outcome_df, outdir / "figs/outcome_distributions.png")

    # ── Regression multiverse ──
    print("Running regression multiverse …", flush=True)
    all_reg = []
    for rep_name, cent_df in cent_dfs.items():
        print(f"  Regressions for: {rep_name} …", flush=True)
        reg_df = run_regression_multiverse(cent_df, outcome_df, rep_name)
        print(f"    -> {len(reg_df)} specifications", flush=True)
        all_reg.append(reg_df)

    reg_all = pd.concat(all_reg, ignore_index=True)
    reg_all.to_csv(outdir / "tables/regression_multiverse.csv", index=False)

    # ── AME summary ──
    ame_summary = (
        reg_all.dropna(subset=["ame"])
        .groupby(["centrality", "outcome", "model"])
        .agg(
            mean_ame=("ame", "mean"),
            median_ame=("ame", "median"),
            pct_significant=("significant", "mean"),
            n_universes=("ame", "count"),
        )
        .reset_index()
        .sort_values("median_ame", ascending=False)
    )
    ame_summary.to_csv(outdir / "tables/ame_summary.csv", index=False)

    # ── Figures ──
    print("Plotting specification curve …", flush=True)
    plot_specification_curve(reg_all, outdir / "figs/ame_specification_curve.png")
    plot_ame_by_centrality(reg_all, outdir / "figs/ame_by_centrality.png")

    # ── LaTeX tables ──
    print("Writing LaTeX tables …", flush=True)

    df_to_latex(
        stats_df,
        caption="Summary of network representations in the Karnataka multiverse.",
        label="tab:universe_summary",
        outpath=outdir / "tables/universe_summary.tex",
    )

    pr_stab = stab_df[stab_df["measure"] == "pagerank"].head(20).copy()
    pr_stab = pr_stab[["node", "top_k_count", "top_k_freq", "n_universes"]].rename(columns={
        "node": "Node",
        "top_k_count": "Top-20 Count",
        "top_k_freq":  "Top-20 Freq.",
        "n_universes": "N Universes",
    })
    df_to_latex(
        pr_stab,
        caption="Most stable nodes under PageRank across representations (Top-20 frequency).",
        label="tab:pagerank_stability",
        outpath=outdir / "tables/pagerank_stability.tex",
    )

    ame_top = ame_summary.head(30).copy()
    ame_top = ame_top[["centrality", "outcome", "model",
                        "mean_ame", "pct_significant", "n_universes"]].rename(columns={
        "centrality":      "Centrality",
        "outcome":         "Outcome",
        "model":           "Model",
        "mean_ame":        "Mean AME",
        "pct_significant": "Frac.\ Sig.",
        "n_universes":     "N",
    })
    df_to_latex(
        ame_top,
        caption="Top 30 universe-level AME summaries (sorted by median AME).",
        label="tab:ame_summary",
        outpath=outdir / "tables/ame_summary.tex",
    )

    print(f"\nDone. Outputs written to: {outdir}", flush=True)
    print(f"  Tables: {sorted([p.name for p in (outdir/'tables').glob('*.*')])}", flush=True)
    print(f"  Figs:   {sorted([p.name for p in (outdir/'figs').glob('*.png')])}", flush=True)


if __name__ == "__main__":
    main()
