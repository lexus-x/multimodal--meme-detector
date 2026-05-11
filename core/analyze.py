"""
Multimodal Meme Offensive Content Detection — Analysis & Visualization

Generates publication-quality plots for model comparison and error analysis.

Usage:
    python -m core.analyze --runs checkpoints/early_bilstm checkpoints/cross_attention_bilstm
    python -m core.analyze --runs checkpoints/cross_attention_bilstm --single
"""

import os
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    average_precision_score, classification_report,
)

# ── Style ─────────────────────────────────────────────────────────────

PALETTE = sns.color_palette("husl", 8)
sns.set_theme(style="whitegrid", font_scale=1.1)


def load_run(run_dir: str) -> dict:
    """Load a training run's history + args."""
    history_path = os.path.join(run_dir, "history.json")
    args_path = os.path.join(run_dir, "args.json")

    with open(history_path) as f:
        history = json.load(f)
    with open(args_path) as f:
        args = json.load(f)

    return {"dir": run_dir, "history": history, "args": args}


# ── Training Curves ───────────────────────────────────────────────────

def plot_training_curves(runs: list[dict], output_dir: str):
    """Plot loss and F1 curves for multiple runs."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, run in enumerate(runs):
        label = run["args"].get("fusion", "?") + " + " + run["args"].get("text_encoder", "?")
        epochs = [h["epoch"] for h in run["history"]]
        train_loss = [h["train"]["loss"] for h in run["history"]]
        val_loss = [h["val"]["loss"] for h in run["history"]]
        val_f1 = [h["val"]["f1"] for h in run["history"]]
        val_p = [h["val"]["precision"] for h in run["history"]]
        val_r = [h["val"]["recall"] for h in run["history"]]

        color = PALETTE[i % len(PALETTE)]

        axes[0].plot(epochs, train_loss, "--", color=color, alpha=0.7, label=f"{label} (train)")
        axes[0].plot(epochs, val_loss, "-", color=color, label=f"{label} (val)")
        axes[1].plot(epochs, val_f1, "-o", color=color, markersize=4, label=label)
        axes[2].plot(epochs, val_p, "--", color=color, alpha=0.7, label=f"{label} (P)")
        axes[2].plot(epochs, val_r, "-", color=color, label=f"{label} (R)")

    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Training & Validation Loss")
    axes[1].set(xlabel="Epoch", ylabel="F1 Score", title="Validation F1")
    axes[2].set(xlabel="Epoch", ylabel="Score", title="Precision & Recall")

    for ax in axes:
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")


# ── Confusion Matrix ──────────────────────────────────────────────────

def plot_confusion_matrix(targets, preds, title: str, output_path: str):
    cm = confusion_matrix(targets, preds)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=["Non-offensive", "Offensive"],
                yticklabels=["Non-offensive", "Offensive"])
    axes[0].set(xlabel="Predicted", ylabel="Actual", title=f"{title} (counts)")

    # Percentages
    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues", ax=axes[1],
                xticklabels=["Non-offensive", "Offensive"],
                yticklabels=["Non-offensive", "Offensive"])
    axes[1].set(xlabel="Predicted", ylabel="Actual", title=f"{title} (%)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {output_path}")


# ── ROC & PR Curves ──────────────────────────────────────────────────

def plot_roc_pr_curves(all_results: dict, output_dir: str):
    """Plot ROC and Precision-Recall curves for multiple models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for i, (name, res) in enumerate(all_results.items()):
        targets = res["targets"]
        probs = res["probs"]
        color = PALETTE[i % len(PALETTE)]

        # ROC
        fpr, tpr, _ = roc_curve(targets, probs)
        roc_auc = auc(fpr, tpr)
        axes[0].plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={roc_auc:.3f})")

        # PR
        prec, rec, _ = precision_recall_curve(targets, probs)
        ap = average_precision_score(targets, probs)
        axes[1].plot(rec, prec, color=color, lw=2, label=f"{name} (AP={ap:.3f})")

    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3)
    axes[0].set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curve")
    axes[0].legend(fontsize=9)

    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curve")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(output_dir, "roc_pr_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")


# ── Confidence Distribution ───────────────────────────────────────────

def plot_confidence_distribution(targets, probs, title: str, output_path: str):
    """Plot probability distribution for correct vs incorrect predictions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # By class
    for cls, label, color in [(0, "Non-offensive", PALETTE[0]), (1, "Offensive", PALETTE[3])]:
        mask = targets == cls
        axes[0].hist(probs[mask], bins=30, alpha=0.6, color=color, label=label, density=True)
    axes[0].axvline(0.5, color="red", linestyle="--", alpha=0.5)
    axes[0].set(xlabel="Predicted Probability", ylabel="Density",
                title=f"{title} — Probability Distribution by Class")
    axes[0].legend()

    # Correct vs incorrect
    correct = (probs > 0.5).astype(int) == targets
    axes[1].hist(probs[correct], bins=20, alpha=0.6, color="green", label="Correct", density=True)
    axes[1].hist(probs[~correct], bins=20, alpha=0.6, color="red", label="Incorrect", density=True)
    axes[1].axvline(0.5, color="black", linestyle="--", alpha=0.5)
    axes[1].set(xlabel="Predicted Probability", ylabel="Density",
                title=f"{title} — Correct vs Incorrect")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {output_path}")


# ── Model Comparison Bar Chart ────────────────────────────────────────

def plot_model_comparison(all_results: dict, output_dir: str):
    """Bar chart comparing P/R/F1 across models."""
    models = list(all_results.keys())
    metrics = {"Precision": [], "Recall": [], "F1": []}

    for name in models:
        r = all_results[name]
        t, p = r["targets"], r["preds"]
        from sklearn.metrics import precision_score, recall_score, f1_score
        metrics["Precision"].append(precision_score(t, p, zero_division=0))
        metrics["Recall"].append(recall_score(t, p, zero_division=0))
        metrics["F1"].append(f1_score(t, p, zero_division=0))

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(8, len(models) * 2), 6))
    for i, (metric, values) in enumerate(metrics.items()):
        bars = ax.bar(x + i * width, values, width, label=metric, color=PALETTE[i])
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set(xticks=x + width, xticklabels=models, ylabel="Score",
           title="Model Comparison — Offensive Meme Detection")
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(output_dir, "model_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze training runs")
    parser.add_argument("--runs", nargs="+", required=True, help="Run directories")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--single", action="store_true", help="Detailed analysis for single run")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    runs = [load_run(r) for r in args.runs]
    print(f"\nAnalyzing {len(runs)} run(s)...\n")

    # Training curves
    plot_training_curves(runs, args.output_dir)

    # For single-run deep analysis
    if args.single or len(runs) == 1:
        run = runs[0]
        name = run["args"].get("fusion", "model") + "_" + run["args"].get("text_encoder", "")

        # Get final epoch predictions (from history — need to re-run evaluate for full preds)
        final = run["history"][-1]["val"]
        print(f"\n  Best epoch: {final.get('epoch', run['history'][-1]['epoch'])}")
        print(f"  Final F1: {final.get('f1', 'N/A')}")

    # Summary table
    print(f"\n{'─' * 70}")
    print(f"{'Model':<30} {'F1':>6} {'P':>6} {'R':>6} {'Best Ep':>8}")
    print(f"{'─' * 70}")
    for run in runs:
        name = run["args"].get("fusion", "?") + " + " + run["args"].get("text_encoder", "?")
        best = max(run["history"], key=lambda h: h["val"]["f1"])
        print(f"{name:<30} {best['val']['f1']:6.3f} "
              f"{best['val']['precision']:6.3f} {best['val']['recall']:6.3f} "
              f"{best['epoch']:8d}")
    print(f"{'─' * 70}")


if __name__ == "__main__":
    main()
