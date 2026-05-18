#!/usr/bin/env python3
"""
Export the Karnataka loan nomination graph as a CSV edge list for use by R.
Usage: python3 python/export_edgelist.py --graph data/loan_nomination_graph.pkl \
           --out data/loan_nomination_edgelist.csv
"""
import argparse, pickle
import networkx as nx
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--graph", default="data/loan_nomination_graph.pkl")
ap.add_argument("--out",   default="data/loan_nomination_edgelist.csv")
args = ap.parse_args()

with open(args.graph, "rb") as f:
    G = pickle.load(f)

rows = []
for u, v, d in G.edges(data=True):
    rows.append({"from": str(u), "to": str(v), "weight": d.get("weight", 1.0)})

df = pd.DataFrame(rows)
df.to_csv(args.out, index=False)
print(f"Exported {len(df)} edges to {args.out}")
print(f"Sample:\n{df.head()}")
