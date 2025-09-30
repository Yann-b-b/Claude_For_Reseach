#!/usr/bin/env python3
"""
Generate plots showing average, median, minimum, and maximum lengths of
DNA and protein sequences from an antimicrobial training CSV.

Output:
  - plots/dna_length_stats.png
  - plots/protein_length_stats.png

The script will look for the dataset in the following order:
  1) combined_amp_training_data.csv
  2) antimicrobial_training_data.csv
  3) Dataset_and_train_sequence/antimicrobial_training_data.csv

Usage:
  python scripts/plot_length_stats.py
"""

import os
from typing import Optional, Dict

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def find_dataset_path() -> Optional[str]:
    candidates = [
        "combined_amp_training_data.csv",
        "antimicrobial_training_data.csv",
        os.path.join("Dataset_and_train_sequence", "antimicrobial_training_data.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def compute_stats(series: pd.Series) -> Dict[str, float]:
    s = series.dropna().astype(str).str.len()
    if s.empty:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(s.mean()),
        "median": float(s.median()),
        "min": float(s.min()),
        "max": float(s.max()),
    }


def plot_stats(title: str, stats: Dict[str, float], out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # Order for display
    keys = ["min", "median", "mean", "max"]
    values = [stats[k] for k in keys]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(keys, values, color=["#6baed6", "#9ecae1", "#c6dbef", "#4292c6"])
    plt.title(title)
    plt.ylabel("Length (nt or aa)")
    plt.grid(axis="y", linestyle=":", alpha=0.5)

    # Add value labels
    for b, v in zip(bars, values):
        plt.text(b.get_x() + b.get_width()/2, b.get_height(), f"{v:.0f}",
                 ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    ds_path = find_dataset_path()
    if not ds_path:
        raise SystemExit("No dataset found. Expected one of the known CSV paths.")

    print(f"Loading dataset from {ds_path}...")
    df = pd.read_csv(ds_path)

    if not {"dna_sequence", "protein_sequence"}.issubset(df.columns):
        missing = {"dna_sequence", "protein_sequence"} - set(df.columns)
        raise SystemExit(f"Missing required columns in CSV: {missing}")

    dna_stats = compute_stats(df["dna_sequence"])
    pep_stats = compute_stats(df["protein_sequence"])

    print("DNA length stats:", dna_stats)
    print("Protein length stats:", pep_stats)

    plot_stats("DNA sequence length stats", dna_stats, os.path.join("plots", "dna_length_stats.png"))
    plot_stats("Protein sequence length stats", pep_stats, os.path.join("plots", "protein_length_stats.png"))

    print("Saved plots to ./plots/")


if __name__ == "__main__":
    main()


