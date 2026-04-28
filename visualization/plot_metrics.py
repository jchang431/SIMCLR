"""
visualization/plot_metrics.py

Reads the JSON metric files written by SimCLRTrainer and LinearEvalTrainer
and produces publication-quality plots saved under `./plots/`.

Usage (from repo root):
    python visualization/plot_metrics.py

Optional flags:
    --pretrain   path/to/pretrain_metrics.json   (default: checkpoints/pretrain_metrics.json)
    --linear     path/to/linear_metrics.json     (default: checkpoints/linear_metrics.json)
    --test       path/to/test_results.json       (default: checkpoints/test_results.json)
    --out        output directory for PNGs       (default: plots/)
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Style
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.35,
    }
)

PRETRAIN_COLOR = "#2196F3"   # blue
LINEAR_LOSS_COLOR = "#E91E63"  # pink
LINEAR_ACC_COLOR = "#4CAF50"   # green
BAR_PALETTE = plt.cm.tab10.colors


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_json(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Metrics file not found: {path}\n"
            "Run the trainer first, or pass a custom path with the CLI flag."
        )
    with open(path) as f:
        return json.load(f)


def _save(fig: plt.Figure, out_dir: str, name: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    dest = os.path.join(out_dir, name)
    fig.savefig(dest, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {dest}")
    return dest


# ─────────────────────────────────────────────────────────────────────────────
# Plot: SimCLR pre-training loss curve
# ─────────────────────────────────────────────────────────────────────────────

def plot_pretrain(metrics_path: str, out_dir: str) -> str:
    data = _load_json(metrics_path)
    losses = data["epoch_losses"]
    n = len(losses)
    epochs = list(range(1, n + 1))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, losses, color=PRETRAIN_COLOR, linewidth=2, marker="o", markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("NT-Xent Loss (avg per epoch)")
    ax.set_title(
        f"SimCLR Pre-training Loss — {data.get('dataset', 'CIFAR-10')}\n"
        f"({n} epochs, {data.get('total_time_s', '?'):.0f}s total)"
    )
    ax.set_xlim(0.5, n + 0.5)

    # Force integer-only ticks; step up to 5 if there are many epochs
    step = max(1, n // 10)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(step))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))

    # Mark best (lowest) epoch
    best_epoch = int(np.argmin(losses)) + 1
    best_val = min(losses)
    y_range = max(losses) - min(losses) or 1
    ax.annotate(
        f"best: {best_val:.3f}",
        xy=(best_epoch, best_val),
        xytext=(best_epoch - max(1, n * 0.12), best_val + y_range * 0.1),
        arrowprops=dict(arrowstyle="->", color="gray"),
        fontsize=9,
        color="gray",
    )

    fig.tight_layout()
    return _save(fig, out_dir, "pretrain_loss.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot: Linear evaluation — loss + accuracy
# ─────────────────────────────────────────────────────────────────────────────

def plot_linear(metrics_path: str, out_dir: str) -> str:
    data = _load_json(metrics_path)
    losses = data["epoch_losses"]
    accs = data["epoch_accs"]
    n = len(losses)
    epochs = list(range(1, n + 1))
    step = max(1, n // 10)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(
        f"Linear Evaluation — {data.get('dataset', 'CIFAR-10')}   "
        f"({n} epochs, {data.get('total_time_s', '?'):.0f}s total)",
        fontsize=12,
    )

    # ── Loss ──
    ax1.plot(epochs, losses, color=LINEAR_LOSS_COLOR, linewidth=2, marker="o", markersize=4)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss (avg per epoch)")
    ax1.set_title("Training Loss")
    ax1.set_xlim(0.5, n + 0.5)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(step))
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))

    # ── Accuracy ──
    ax2.plot(epochs, accs, color=LINEAR_ACC_COLOR, linewidth=2, marker="o", markersize=4)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Training Accuracy")
    ax2.set_xlim(0.5, n + 0.5)
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(step))
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))

    best_epoch = int(np.argmax(accs)) + 1
    best_acc = max(accs)
    y_range = max(accs) - min(accs) or 1
    ax2.annotate(
        f"best: {best_acc:.1f}%",
        xy=(best_epoch, best_acc),
        xytext=(best_epoch - max(1, n * 0.12), best_acc - y_range * 0.15),
        arrowprops=dict(arrowstyle="->", color="gray"),
        fontsize=9,
        color="gray",
    )

    fig.tight_layout()
    return _save(fig, out_dir, "linear_eval.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot: Per-class test accuracy bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_test_accuracy(test_path: str, out_dir: str) -> str:
    data = _load_json(test_path)
    per_class = data["per_class_accuracy"]
    avg_acc = data["avg_accuracy"]

    classes = list(per_class.keys())
    values = [per_class[c] if per_class[c] is not None else 0.0 for c in classes]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(classes, values, color=BAR_PALETTE[: len(classes)], edgecolor="white", linewidth=0.6)

    # Average line
    ax.axhline(avg_acc, color="black", linestyle="--", linewidth=1.2, label=f"Average: {avg_acc:.1f}%")
    ax.legend(fontsize=9)

    # Value labels on bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.4,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

    ax.set_xlabel("Class")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Per-class Test Accuracy — CIFAR-10")
    ax.set_ylim(0, max(values) * 1.15)
    plt.xticks(rotation=20, ha="right")

    fig.tight_layout()
    return _save(fig, out_dir, "test_per_class_accuracy.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot: Pretrain + linear loss on one combined axis (for easy comparison)
# ─────────────────────────────────────────────────────────────────────────────

def plot_combined_loss(pretrain_path: str, linear_path: str, out_dir: str) -> str:
    pre = _load_json(pretrain_path)
    lin = _load_json(linear_path)

    pre_n = len(pre["epoch_losses"])
    lin_n = len(lin["epoch_losses"])
    pre_epochs = list(range(1, pre_n + 1))
    lin_epochs = list(range(1, lin_n + 1))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(pre_epochs, pre["epoch_losses"], color=PRETRAIN_COLOR, linewidth=2,
            label="Pre-training (NT-Xent)", marker="o", markersize=3)
    ax2 = ax.twinx()
    ax2.plot(lin_epochs, lin["epoch_losses"], color=LINEAR_LOSS_COLOR, linewidth=2,
             label="Linear Eval (CE)", marker="s", markersize=3)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("NT-Xent Loss (avg per epoch)", color=PRETRAIN_COLOR)
    ax2.set_ylabel("Cross-Entropy Loss (avg per epoch)", color=LINEAR_LOSS_COLOR)
    ax.tick_params(axis="y", labelcolor=PRETRAIN_COLOR)
    ax2.tick_params(axis="y", labelcolor=LINEAR_LOSS_COLOR)
    ax2.spines["right"].set_visible(True)

    # Integer x-ticks (use the larger of the two epoch counts for spacing)
    step = max(1, max(pre_n, lin_n) // 10)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(step))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)

    ax.set_title("Pre-training vs. Linear Eval Loss")
    fig.tight_layout()
    return _save(fig, out_dir, "combined_loss.png")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(description="Plot SimCLR training metrics")
    parser.add_argument("--pretrain", default="checkpoints/pretrain_metrics.json",
                        help="Path to pretrain_metrics.json")
    parser.add_argument("--linear", default="checkpoints/linear_metrics.json",
                        help="Path to linear_metrics.json")
    parser.add_argument("--test", default="checkpoints/test_results.json",
                        help="Path to test_results.json")
    parser.add_argument("--out", default="plots",
                        help="Output directory for PNG files")
    return parser.parse_args()


def main():
    args = _parse_args()

    generated = []

    if os.path.exists(args.pretrain):
        generated.append(plot_pretrain(args.pretrain, args.out))
    else:
        print(f"[skip] pretrain metrics not found at {args.pretrain}")

    if os.path.exists(args.linear):
        generated.append(plot_linear(args.linear, args.out))
    else:
        print(f"[skip] linear metrics not found at {args.linear}")

    if os.path.exists(args.test):
        generated.append(plot_test_accuracy(args.test, args.out))
    else:
        print(f"[skip] test results not found at {args.test}")

    if os.path.exists(args.pretrain) and os.path.exists(args.linear):
        generated.append(plot_combined_loss(args.pretrain, args.linear, args.out))

    if generated:
        print(f"\nAll plots saved to: {os.path.abspath(args.out)}/")
    else:
        print(
            "\nNo metric files found. Run pretraining and/or linear eval first, "
            "then re-run this script."
        )


if __name__ == "__main__":
    main()
