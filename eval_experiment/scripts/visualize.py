"""Generate heatmap from deltas.csv.

Usage:
    cd eval_experiment/
    python scripts/visualize.py
    python scripts/visualize.py --scores results/scores.csv --deltas results/deltas.csv
"""

import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

matplotlib.rcParams["font.family"] = "DejaVu Sans"


BENCH_DISPLAY = {
    "MATH-500": "MATH-500\n(on-target)",
    "GSM8K":    "GSM8K",
    "MBPP":     "MBPP",
    "IFEval":   "IFEval",
    "MMLU-Pro": "MMLU-Pro",
    "BBH":      "BBH",
    "MGSM":     "MGSM",
    "C-Eval":   "C-Eval",
}


def load_data(scores_path: str, deltas_path: str):
    scores = pd.read_csv(scores_path, index_col=0)
    deltas = pd.read_csv(deltas_path, index_col=0)
    return scores, deltas


def build_heatmap_data(deltas: pd.DataFrame):
    """Reorder rows: layer_0..27 (sorted), then full_rlvr. Baseline is zero line."""
    layer_rows = sorted(
        [r for r in deltas.index if r.startswith("layer_")],
        key=lambda x: int(x.split("_")[1])
    )
    order = layer_rows + ["full_rlvr"]
    present = [r for r in order if r in deltas.index]
    return deltas.loc[present]


def plot_heatmap(
    deltas: pd.DataFrame,
    scores: pd.DataFrame,
    out_path: str = "results/heatmap.png",
    figsize: tuple = (14, 12),
):
    data = build_heatmap_data(deltas) * 100  # convert to percentage points

    n_rows, n_cols = data.shape
    fig, ax = plt.subplots(figsize=figsize)

    # Color scale: symmetric around 0, max abs determines range
    vmax = max(abs(data.values[~np.isnan(data.values)]).max(), 0.1)
    vmin = -vmax

    im = ax.imshow(data.values, cmap="RdYlGn", aspect="auto",
                   vmin=vmin, vmax=vmax, interpolation="nearest")

    # Axes labels
    col_labels = [BENCH_DISPLAY.get(c, c) for c in data.columns]
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=9, ha="center")
    ax.set_yticks(range(n_rows))

    row_labels = []
    for label in data.index:
        if label == "full_rlvr":
            row_labels.append("Full RLVR")
        elif label.startswith("layer_"):
            row_labels.append(f"Layer {label.split('_')[1]}")
        else:
            row_labels.append(label)
    ax.set_yticklabels(row_labels, fontsize=8)

    # Cell annotations: show delta value in pp
    for i in range(n_rows):
        for j in range(n_cols):
            val = data.values[i, j]
            if np.isnan(val):
                text = "N/A"
                color = "gray"
            else:
                text = f"{val:+.1f}"
                color = "black" if abs(val) < vmax * 0.6 else "white"
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=7, color=color, fontweight="normal")

    # Separator line before "Full RLVR" row
    full_idx = list(data.index).index("full_rlvr") if "full_rlvr" in data.index else None
    if full_idx is not None:
        ax.axhline(full_idx - 0.5, color="black", linewidth=2)

    # Vertical separator after MATH-500 (first column = on-target)
    ax.axvline(0.5, color="navy", linewidth=1.5, linestyle="--", alpha=0.6)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("Δ vs baseline (pp)", fontsize=10)

    # Baseline score annotation in top-right
    if "baseline" in scores.index:
        bl = scores.loc["baseline"] * 100
        baseline_txt = "Baseline: " + ", ".join(
            f"{c}={bl[c]:.1f}" for c in scores.columns if c in bl.index
        )
        fig.text(0.01, 0.01, baseline_txt, fontsize=7,
                 color="gray", ha="left", va="bottom")

    ax.set_title(
        "Single-Layer RLVR Off-Target Capability Evaluation\n"
        "Δ vs Qwen3-1.7B-Instruct baseline (percentage points)\n"
        "Green = improvement, Red = degradation",
        fontsize=12, fontweight="bold", pad=15,
    )
    ax.set_xlabel("Benchmark", fontsize=11, labelpad=8)
    ax.set_ylabel("Checkpoint (layer trained)", fontsize=11, labelpad=8)

    # Legend patches
    patches = [
        mpatches.Patch(color="green",  label="Improvement vs baseline"),
        mpatches.Patch(color="red",    label="Degradation vs baseline"),
        mpatches.Patch(color="white",  edgecolor="black", label="No change"),
    ]
    ax.legend(handles=patches, loc="upper right",
              bbox_to_anchor=(1.0, -0.06), ncol=3, fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Heatmap saved: {out_path}")
    plt.close()


def plot_pareto(deltas: pd.DataFrame, out_path: str = "results/pareto.png"):
    """Scatter plot: MATH-500 gain vs mean off-target change per layer."""
    data = deltas * 100

    math_col = "MATH-500"
    off_cols = [c for c in data.columns if c != math_col]

    layer_rows = sorted(
        [r for r in data.index if r.startswith("layer_")],
        key=lambda x: int(x.split("_")[1])
    )

    if not layer_rows or math_col not in data.columns:
        print("[skip] Pareto plot: insufficient data")
        return

    xs = data.loc[layer_rows, off_cols].mean(axis=1)
    ys = data.loc[layer_rows, math_col]

    fig, ax = plt.subplots(figsize=(10, 7))

    scatter = ax.scatter(xs, ys, c=range(len(layer_rows)),
                         cmap="viridis", s=80, zorder=3)

    for i, label in enumerate(layer_rows):
        layer_num = label.split("_")[1]
        ax.annotate(layer_num, (xs.iloc[i], ys.iloc[i]),
                    textcoords="offset points", xytext=(4, 4), fontsize=7)

    # Full RLVR point
    if "full_rlvr" in data.index:
        fx = data.loc["full_rlvr", off_cols].mean()
        fy = data.loc["full_rlvr", math_col]
        ax.scatter([fx], [fy], marker="*", s=200, color="red",
                   zorder=4, label="Full RLVR")
        ax.annotate("Full RLVR", (fx, fy), textcoords="offset points",
                    xytext=(6, 4), fontsize=8, color="red")

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")

    ax.set_xlabel("Mean off-target Δ (pp, 7 benchmarks)", fontsize=11)
    ax.set_ylabel("MATH-500 Δ (pp, on-target)", fontsize=11)
    ax.set_title("Pareto Frontier: MATH gain vs Off-target change\n"
                 "Top-right = high math gain, low off-target degradation", fontsize=11)

    cbar = fig.colorbar(scatter, ax=ax, fraction=0.02)
    cbar.set_label("Layer index", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Pareto plot saved: {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", default="results/scores.csv")
    parser.add_argument("--deltas", default="results/deltas.csv")
    parser.add_argument("--heatmap-out", default="results/heatmap.png")
    parser.add_argument("--pareto-out",  default="results/pareto.png")
    args = parser.parse_args()

    for p in [args.scores, args.deltas]:
        if not os.path.exists(p):
            print(f"[error] File not found: {p}")
            print("Run aggregate.py first.")
            return

    scores, deltas = load_data(args.scores, args.deltas)
    print(f"Loaded {len(deltas)} checkpoints × {len(deltas.columns)} benchmarks")

    plot_heatmap(deltas, scores, out_path=args.heatmap_out)
    plot_pareto(deltas, out_path=args.pareto_out)


if __name__ == "__main__":
    main()
