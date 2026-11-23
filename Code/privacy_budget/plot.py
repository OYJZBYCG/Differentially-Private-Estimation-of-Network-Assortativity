"""Plot Figs. 2, 5 and 6 from the supplementary experiment summaries."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATASETS = [
    ("BA_m10", r"BA ($m=10$)"), ("BA_m50", r"BA ($m=50$)"),
    ("BA_m100", r"BA ($m=100$)"), ("Facebook", "Facebook"),
    ("Deezer", "Deezer"), ("GitHub", "GitHub"),
]
METHODS = [
    ("local", r"$\mathbf{Local}_{r}^{(1)}$", "#F2AA3B", "s"),
    ("ikos_shuffle", r"$\mathbf{Shuffle}_{r}^{(1)}$", "#28B9DF", "^"),
    ("decentral", r"$\mathbf{Local}_{r}^{(2)}$", "#83CC91", "o"),
]


def draw(ax, data, x, metric, ablation):
    variants = ("plugin", "corrected") if ablation else ("corrected",)
    for variant in variants:
        for method, label, color, marker in METHODS:
            points = data[(data.variant == variant) & (data.mechanism == method)].sort_values(x)
            if points.empty or points[x].duplicated().any():
                raise ValueError(f"Missing or duplicate points for {method}/{variant}")
            text = label + (" (plug-in)" if variant == "plugin" else "")
            ax.plot(points[x], points[metric], label=text, color=color, marker=marker,
                    linestyle="--" if variant == "plugin" else "-", lw=2.2, ms=6.5)
    ax.set_yscale("log")
    ax.set_ylabel(metric.upper())
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=5))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(axis="y", linestyle="--", color="#d6d6d6")
    for spine in ax.spines.values():
        spine.set_color("#888888")


def plot(data, datasets, output, ablation=False):
    fig, axes = plt.subplots(3, 4, figsize=(18, 9.6))
    fig.subplots_adjust(left=.055, right=.995, bottom=.11, top=.91, wspace=.34, hspace=.69)
    x = "epsilon" if ablation else "degree_share"
    for row in range(3):
        for col in range(4):
            ax = axes[row, col]
            dataset, label = datasets[2 * row + col % 2 if ablation else row]
            subset = data[data.dataset == dataset]
            if not ablation:
                eps = (.5, 1)[col % 2]
                subset = subset[subset.epsilon == eps]
                label += rf", $\varepsilon={eps:g}$"
            draw(ax, subset, x, "mse" if col < 2 else "mare", ablation)
            ax.set_xlabel(r"$\varepsilon$" if ablation else r"$\rho=\varepsilon_d/\varepsilon$")
            ax.set_xticks([.1, .5, 1, 1.5, 2, 2.5, 3] if ablation else [.1, .3, .5, .7, .9])
            ax.annotate(f"({chr(97 + 4 * row + col)}) {label}", (.5, 0),
                        xycoords="axes fraction", xytext=(0, -50),
                        textcoords="offset points", ha="center", va="top")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if ablation:
        handles, labels = handles[3:] + handles[:3], labels[3:] + labels[:3]
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False)
    fig.savefig(output, bbox_inches="tight", pad_inches=.08)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--correction", type=Path, required=True, help="Ablation run summary.csv")
    parser.add_argument("--allocation", type=Path, required=True, help="Allocation run summary.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    plt.rcParams.update({"font.family": "serif", "font.size": 15,
                         "mathtext.fontset": "stix", "pdf.fonttype": 42})
    plot(pd.read_csv(args.correction), DATASETS,
         args.output / "privacy_budget_six_curves_3x4.pdf", ablation=True)
    allocation = pd.read_csv(args.allocation)
    plot(allocation, DATASETS[:3], args.output / "allocation_BA_3x4.pdf")
    plot(allocation, DATASETS[3:], args.output / "allocation_real_3x4.pdf")


if __name__ == "__main__":
    main()
