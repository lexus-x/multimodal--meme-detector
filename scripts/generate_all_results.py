"""
Generate all evaluation results and publication-quality figures.
Evaluates all trained fusion checkpoints on the test set and produces:
  - Confusion matrices
  - Training curves
  - Fusion comparison bar chart
  - Architecture diagram
  - Example meme inference grid
  - ROC curves
  - Dataset distribution
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, f1_score, precision_score, recall_score, accuracy_score
)
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.dataset import MultiOFFDataset, load_glove_embeddings
from core.models import MultimodalClassifier

# ── Config ────────────────────────────────────────────────────────
OUT = "data/output/figures"
CKPT_DIR = "research/checkpoints"
TEST_CSV = "data/processed/test.csv"
IMG_DIR  = "data/processed/images"
GLOVE    = "data/glove.6B/glove.6B.50d.txt"

os.makedirs(OUT, exist_ok=True)
sns.set_theme(style="whitegrid", font_scale=1.1)
COLORS = ["#1E88E5", "#E53935", "#43A047", "#FB8C00"]

RUNS = [
    ("early_bilstm_vgg16",           "Early Fusion"),
    ("cross_attention_bilstm_vgg16", "Cross-Attention"),
    ("gated_bilstm_vgg16",           "Gated Fusion"),
    ("bilinear_bilstm_vgg16",        "Bilinear Fusion"),
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab, embeddings = load_glove_embeddings(GLOVE, 50)
test_ds = MultiOFFDataset(TEST_CSV, IMG_DIR, vocab)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

# ── Helper: load & evaluate ──────────────────────────────────────
@torch.no_grad()
def eval_checkpoint(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    model = MultimodalClassifier(
        embedding_matrix=embeddings,
        text_hidden_dim=args.get("text_hidden", 128),
        text_encoder=args.get("text_encoder", "bilstm"),
        img_hidden_dim=args.get("img_hidden", 256),
        img_backbone=args.get("img_backbone", "vgg16"),
        fusion_type=args.get("fusion", "early"),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_probs, all_preds, all_targets = [], [], []
    for batch in test_loader:
        imgs = batch["image"].to(device)
        txts = batch["text"].to(device)
        logits = model(txts, imgs)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs)
        all_preds.extend((probs > 0.5).astype(int))
        all_targets.extend(batch["label"].numpy().astype(int))

    return np.array(all_targets), np.array(all_preds), np.array(all_probs)


# ═══════════════════════════════════════════════════════════════════
#  1. EVALUATE ALL MODELS
# ═══════════════════════════════════════════════════════════════════
print("=" * 60)
print("  EVALUATING ALL MODELS ON TEST SET (148 samples)")
print("=" * 60)

results = {}
for run_name, label in RUNS:
    ckpt = os.path.join(CKPT_DIR, run_name, "best_model.pth")
    if not os.path.exists(ckpt):
        print(f"  SKIP {label} — no checkpoint")
        continue
    targets, preds, probs = eval_checkpoint(ckpt)
    acc = accuracy_score(targets, preds)
    p   = precision_score(targets, preds, zero_division=0)
    r   = recall_score(targets, preds, zero_division=0)
    f1  = f1_score(targets, preds, zero_division=0)
    try:
        auc = roc_auc_score(targets, probs)
    except ValueError:
        auc = 0.0
    results[run_name] = {
        "label": label, "targets": targets, "preds": preds, "probs": probs,
        "acc": acc, "precision": p, "recall": r, "f1": f1, "auc": auc,
    }
    print(f"  {label:25s}  Acc={acc:.3f}  P={p:.3f}  R={r:.3f}  F1={f1:.3f}  AUC={auc:.3f}")

# Find best
best_key = max(results, key=lambda k: results[k]["f1"])
print(f"\n  ★ Best model: {results[best_key]['label']} (F1={results[best_key]['f1']:.3f})")


# ═══════════════════════════════════════════════════════════════════
#  2. FUSION COMPARISON BAR CHART
# ═══════════════════════════════════════════════════════════════════
print("\nGenerating Figure: Fusion Comparison...")
fig, ax = plt.subplots(figsize=(10, 6))
labels_list = [results[k]["label"] for k in results]
metrics_names = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
x = np.arange(len(labels_list))
width = 0.15

for i, metric in enumerate(["acc", "precision", "recall", "f1", "auc"]):
    vals = [results[k][metric] for k in results]
    bars = ax.bar(x + i * width, vals, width, label=metrics_names[i], color=COLORS[i % len(COLORS)] if i < 4 else "#9C27B0")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{v:.2f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold")

ax.set_xticks(x + width * 2)
ax.set_xticklabels(labels_list, fontsize=11)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Score")
ax.set_title("Fusion Strategy Comparison — MultiOFF Test Set", fontsize=14, fontweight="bold")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{OUT}/fusion_comparison.png", dpi=300, bbox_inches="tight")
plt.close()


# ═══════════════════════════════════════════════════════════════════
#  3. CONFUSION MATRICES (best model)
# ═══════════════════════════════════════════════════════════════════
print("Generating Figure: Confusion Matrix...")
best = results[best_key]
cm = confusion_matrix(best["targets"], best["preds"])
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Non-offensive", "Offensive"],
            yticklabels=["Non-offensive", "Offensive"], ax=ax,
            annot_kws={"size": 16, "fontweight": "bold"})
ax.set_xlabel("Predicted", fontsize=12)
ax.set_ylabel("True", fontsize=12)
ax.set_title(f"Confusion Matrix — {best['label']} (Test Set)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT}/confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.close()


# ═══════════════════════════════════════════════════════════════════
#  4. ROC CURVES (all models)
# ═══════════════════════════════════════════════════════════════════
print("Generating Figure: ROC Curves...")
fig, ax = plt.subplots(figsize=(7, 6))
for i, (k, r) in enumerate(results.items()):
    fpr, tpr, _ = roc_curve(r["targets"], r["probs"])
    ax.plot(fpr, tpr, color=COLORS[i % len(COLORS)], lw=2,
            label=f"{r['label']} (AUC={r['auc']:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — All Fusion Strategies", fontsize=13, fontweight="bold")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{OUT}/roc_curves.png", dpi=300, bbox_inches="tight")
plt.close()


# ═══════════════════════════════════════════════════════════════════
#  5. TRAINING CURVES (all models)
# ═══════════════════════════════════════════════════════════════════
print("Generating Figure: Training Curves...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for i, (run_name, label) in enumerate(RUNS):
    hist_path = os.path.join(CKPT_DIR, run_name, "history.json")
    if not os.path.exists(hist_path):
        continue
    with open(hist_path) as f:
        hist = json.load(f)
    epochs = [h["epoch"] for h in hist]
    train_loss = [h["train"]["loss"] for h in hist]
    val_loss = [h["val"]["loss"] for h in hist]
    val_f1 = [h["val"]["f1"] for h in hist]

    axes[0].plot(epochs, train_loss, color=COLORS[i], lw=2, linestyle="-", label=f"{label} (train)")
    axes[0].plot(epochs, val_loss, color=COLORS[i], lw=2, linestyle="--", alpha=0.6)
    axes[1].plot(epochs, val_f1, color=COLORS[i], lw=2, marker="o", markersize=4, label=label)

axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].set_title("Training Loss (solid) / Val Loss (dashed)")
axes[0].legend(fontsize=9)
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("F1 Score"); axes[1].set_title("Validation F1 Score")
axes[1].legend(fontsize=9)
plt.suptitle("Training Curves — All Fusion Strategies", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT}/training_curves.png", dpi=300, bbox_inches="tight")
plt.close()


# ═══════════════════════════════════════════════════════════════════
#  6. DATASET DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════
print("Generating Figure: Dataset Distribution...")
import pandas as pd
train_df = pd.read_csv("data/processed/train.csv")
val_df   = pd.read_csv("data/processed/val.csv")
test_df  = pd.read_csv("data/processed/test.csv")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
splits = ["Train", "Validation", "Test"]
counts = [len(train_df), len(val_df), len(test_df)]
axes[0].pie(counts, labels=splits, autopct="%1.0f%%",
            colors=["#43A047", "#FB8C00", "#1E88E5"], startangle=140,
            textprops={"fontsize": 12, "fontweight": "bold"})
axes[0].set_title(f"Dataset Split ({sum(counts)} Total)", fontsize=13, fontweight="bold")

classes = ["Non-offensive", "Offensive"]
for split_name, df, color in [("Train", train_df, "#43A047"), ("Val", val_df, "#FB8C00"), ("Test", test_df, "#1E88E5")]:
    n0 = (df["label"] == 0).sum()
    n1 = (df["label"] == 1).sum()

all_df = pd.concat([train_df, val_df, test_df])
n0 = (all_df["label"] == 0).sum()
n1 = (all_df["label"] == 1).sum()
bars = axes[1].bar(classes, [n0, n1], color=["#B0BEC5", "#E53935"])
for bar, v in zip(bars, [n0, n1]):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, str(v),
                 ha="center", fontweight="bold", fontsize=13)
axes[1].set_title("Class Distribution (All Splits)", fontsize=13, fontweight="bold")
axes[1].set_ylabel("Samples")
plt.suptitle("MultiOFF Dataset Overview", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT}/dataset_distribution.png", dpi=300, bbox_inches="tight")
plt.close()


# ═══════════════════════════════════════════════════════════════════
#  7. ARCHITECTURE DIAGRAM
# ═══════════════════════════════════════════════════════════════════
print("Generating Figure: Architecture Diagram...")
fig, ax = plt.subplots(figsize=(12, 9))
ax.axis("off")

def draw_box(ax, x, y, w, h, text, fc="white", ec="black", fs=10):
    rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                    facecolor=fc, edgecolor=ec, linewidth=2)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs, fontweight="bold")
    return (x + w/2, y), (x + w/2, y + h)

def arrow(ax, s, e):
    ax.annotate("", xy=e, xytext=s, arrowprops=dict(arrowstyle="-|>", lw=2, color="#333"))

# Text branch
ax.text(0.15, 0.96, "TEXT BRANCH", ha="center", fontsize=13, fontweight="bold", color="#1E88E5")
t1b, t1t = draw_box(ax, 0.02, 0.86, 0.26, 0.07, "Input Text\n(Tokenized)", fc="#E3F2FD")
t2b, t2t = draw_box(ax, 0.02, 0.74, 0.26, 0.07, "GloVe Embeddings\n(50d)", fc="#BBDEFB")
t3b, t3t = draw_box(ax, 0.02, 0.62, 0.26, 0.07, "BiLSTM Encoder\n(128 hidden)", fc="#90CAF9")
t4b, t4t = draw_box(ax, 0.02, 0.50, 0.26, 0.07, "Text Features (256d)", fc="#64B5F6")
arrow(ax, t1b, t2t); arrow(ax, t2b, t3t); arrow(ax, t3b, t4t)

# Image branch
ax.text(0.85, 0.96, "IMAGE BRANCH", ha="center", fontsize=13, fontweight="bold", color="#43A047")
i1b, i1t = draw_box(ax, 0.72, 0.86, 0.26, 0.07, "Input Image\n(224×224×3)", fc="#E8F5E9")
i2b, i2t = draw_box(ax, 0.72, 0.74, 0.26, 0.07, "VGG16 Backbone\n(Frozen)", fc="#C8E6C9")
i3b, i3t = draw_box(ax, 0.72, 0.62, 0.26, 0.07, "FC Layers\n(4096→256)", fc="#A5D6A7")
i4b, i4t = draw_box(ax, 0.72, 0.50, 0.26, 0.07, "Image Features (256d)", fc="#81C784")
arrow(ax, i1b, i2t); arrow(ax, i2b, i3t); arrow(ax, i3b, i4t)

# Fusion
fb, ft = draw_box(ax, 0.25, 0.36, 0.50, 0.08, "Multimodal Fusion\n(Early / Cross-Attention / Gated / Bilinear)", fc="#FFF9C4", fs=11)
arrow(ax, t4b, (ft[0] - 0.1, ft[1])); arrow(ax, i4b, (ft[0] + 0.1, ft[1]))

# Classifier
c1b, c1t = draw_box(ax, 0.30, 0.22, 0.40, 0.07, "MLP Classifier\n(256 → 128 → 1)", fc="#FFE0B2")
c2b, c2t = draw_box(ax, 0.30, 0.08, 0.40, 0.07, "Sigmoid → Offensive / Non-offensive", fc="#FFCCBC", fs=11)
arrow(ax, fb, c1t); arrow(ax, c1b, c2t)

plt.title("Multimodal Meme Detection — Architecture", fontsize=16, fontweight="bold", y=1.02)
plt.savefig(f"{OUT}/architecture_diagram.png", dpi=300, bbox_inches="tight")
plt.close()


# ═══════════════════════════════════════════════════════════════════
#  8. EXAMPLE INFERENCE GRID
# ═══════════════════════════════════════════════════════════════════
print("Generating Figure: Inference Grid...")
from datasets import load_dataset
import textwrap

hf = load_dataset("Ibrahim-Alam/multi-modal_offensive_meme")
test_raw = hf["test"]

# Load best model
ckpt_path = os.path.join(CKPT_DIR, best_key, "best_model.pth")
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
args_d = ckpt.get("args", {})
model = MultimodalClassifier(
    embedding_matrix=embeddings,
    text_encoder=args_d.get("text_encoder", "bilstm"),
    fusion_type=args_d.get("fusion", "early"),
).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

samples = []
c0, c1 = 0, 0
for i in range(len(test_raw)):
    raw = test_raw[i]
    proc = test_ds[i]
    label = int(raw["label"])
    if label == 0 and c0 < 5:
        samples.append((raw, proc)); c0 += 1
    elif label == 1 and c1 < 5:
        samples.append((raw, proc)); c1 += 1
    if c0 == 5 and c1 == 5:
        break

fig, axes = plt.subplots(2, 5, figsize=(22, 9))
label_map = {0: "Non-offensive", 1: "Offensive"}
for i, (raw, proc) in enumerate(samples):
    ax = axes[i // 5][i % 5]
    ax.imshow(raw["image"])
    ax.axis("off")
    with torch.no_grad():
        txt = proc["text"].unsqueeze(0).to(device)
        img = proc["image"].unsqueeze(0).to(device)
        prob = torch.sigmoid(model(txt, img)).item()
    pred = 1 if prob > 0.5 else 0
    true = int(raw["label"])
    text_w = "\n".join(textwrap.wrap(str(raw["text"])[:80], 28))
    color = "green" if pred == true else "red"
    ax.set_title(f"{text_w}\nTrue: {label_map[true]} | Pred: {label_map[pred]}\n({prob:.2f})",
                 fontsize=9, color=color, fontweight="bold")

plt.suptitle(f"Inference Results — {results[best_key]['label']} Model", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT}/inference_grid.png", dpi=300, bbox_inches="tight")
plt.close()


# ═══════════════════════════════════════════════════════════════════
#  9. SAVE SUMMARY JSON
# ═══════════════════════════════════════════════════════════════════
summary = {}
for k, r in results.items():
    summary[k] = {
        "label": r["label"], "acc": r["acc"], "precision": r["precision"],
        "recall": r["recall"], "f1": r["f1"], "auc": r["auc"],
        "confusion_matrix": confusion_matrix(r["targets"], r["preds"]).tolist(),
    }
with open(f"{OUT}/results_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n{'='*60}")
print(f"  ALL FIGURES SAVED TO: {OUT}/")
print(f"  Files generated:")
for f in sorted(os.listdir(OUT)):
    print(f"    - {f}")
print(f"{'='*60}")
