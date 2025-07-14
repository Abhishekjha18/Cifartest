#!/usr/bin/env python
#python plot_crvq_results.py --csv crvq_analysis_results.csv --out plots

"""
plot_crvq_results.py
====================
Generate publication-quality PNGs from crvq_analysis_results.csv.

  • accuracy-vs-compression curves (3 flavours)
  • line curves: accuracy vs λ for each m
  • line curves: compression vs λ for each m

The script auto-creates an output folder (default: plots).
"""

import argparse, os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid") # nicer default look


# ---------------------------------------------------------------------
def load_df(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    # Ensure numeric
    for col in df.columns:
        if col not in ("lambda", "m", "d", "e"):
            df[col] = pd.to_numeric(df[col], errors="ignore")
    return df


def ensure_dir(out: str):
    os.makedirs(out, exist_ok=True)


# ---------------------------------------------------------------------
def curve_acc_vs_compression(df: pd.DataFrame, out: str):
    plt.figure(figsize=(10,10))
    plt.plot(df["model_comp"], df["no_ft_acc"], "x", label="No FT", color="crimson")
    plt.plot(df["model_comp"], df["block_ft_acc"], "^", label="Block FT", color="darkorange")
    plt.plot(df["model_comp"], df["e2e_ft_acc"], "o", label="E2E FT", color="steelblue")
    plt.xlabel("Compression ratio (×)")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs. Compression")
    plt.legend()
    plt.tight_layout()
    fn = os.path.join(out, "accuracy_vs_compression.png")
    plt.savefig(fn, dpi=150)
    plt.close()
    print("✓", fn)


# ---------------------------------------------------------------------
def line_vs_lambda(df: pd.DataFrame, y_col: str, y_label: str, fname: str):
    """Plot y_col (accuracy or comp) vs λ for each m on one curve plot."""
    plt.figure(figsize=(10,4))
    for m_val, sub in df.groupby("m"):
        sub = sub.sort_values("lambda")
        plt.plot(sub["lambda"], sub[y_col],
                 marker="o", linewidth=2, label=f"m={m_val}")
    plt.xlabel("λ (critical-channel ratio)")
    plt.ylabel(y_label)
    title = y_label.split("(")[0].strip()
    plt.title(f"{title} vs. λ (per m)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
    print("✓", fname)


# ---------------------------------------------------------------------
def main(csv_path: str, out_dir: str):
    ensure_dir(out_dir)
    df = load_df(csv_path)

    # 1. Accuracy–compression scatter
    curve_acc_vs_compression(df, out_dir)

    # 2. Accuracy (E2E-FT) vs λ, one curve per m
    line_vs_lambda(df,
                   y_col="e2e_ft_acc",
                   y_label="E2E-FT accuracy (%)",
                   fname=os.path.join(out_dir, "E2Eacc_vs_lambda_per_m.png"))

    # 3. Model compression vs λ, one curve per m
    line_vs_lambda(df,
                   y_col="model_comp",
                   y_label="Model compression (×)",
                   fname=os.path.join(out_dir, "compression_vs_lambda_per_m.png"))
    
    #4. Accuracy (NO-FT) vs λ, one curve per m
    line_vs_lambda(df,
                   y_col="no_ft_acc",
                   y_label="NO-FT accuracy (%)",
                   fname=os.path.join(out_dir, "NOFTacc_vs_lambda_per_m.png"))




if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="crvq_analysis_results.csv", help="CSV from analysis.py")
    ap.add_argument("--out", default="plots", help="folder for PNGs")
    args = ap.parse_args()
    main(args.csv, args.out)
