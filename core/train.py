"""
Multimodal Meme Offensive Content Detection — Training

Usage:
    python -m core.train --use_mock --epochs 5
    python -m core.train --epochs 20 --text_encoder bilstm --fusion cross_attention
    python -m core.train --config research/configs/default.yaml
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from tqdm import tqdm

from core.dataset import (
    MultiOFFDataset, load_glove_embeddings, create_mock_dataset, TRAIN_AUGMENT,
)
from core.models import MultimodalClassifier


# ── CLI ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train multimodal offensive meme detector")
    p.add_argument("--config", type=str, default=None, help="YAML config file (overrides CLI)")

    # Data
    p.add_argument("--train_csv", type=str, default="data/processed/train.csv")
    p.add_argument("--val_csv", type=str, default="data/processed/val.csv")
    p.add_argument("--img_dir", type=str, default="data/processed/images")
    p.add_argument("--glove_path", type=str, default="glove.6B/glove.6B.50d.txt")
    p.add_argument("--glove_dim", type=int, default=50)
    p.add_argument("--use_mock", action="store_true", help="Use mock dataset for testing")
    p.add_argument("--augment", action="store_true", help="Data augmentation on training set")

    # Model
    p.add_argument("--text_encoder", type=str, default="bilstm", choices=["lstm", "bilstm", "cnn"])
    p.add_argument("--text_hidden", type=int, default=128)
    p.add_argument("--img_hidden", type=int, default=256)
    p.add_argument("--img_backbone", type=str, default="vgg16", choices=["vgg16", "resnet50"])
    p.add_argument("--fusion", type=str, default="early",
                   choices=["early", "cross_attention", "cross_attention_deep", "gated", "bilinear"])

    # Training
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--patience", type=int, default=5, help="Early stopping patience")

    # Output
    p.add_argument("--output_dir", type=str, default="checkpoints")
    p.add_argument("--run_name", type=str, default=None, help="Name for this training run")

    return p.parse_args()


def load_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


# ── Training ──────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_preds, all_targets = [], []

    for batch in tqdm(loader, desc="  Train", leave=False, ncols=80):
        images = batch["image"].to(device)
        texts = batch["text"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(texts, images)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(labels.cpu().numpy())

    n = len(loader.dataset)
    return {
        "loss": total_loss / n,
        "accuracy": accuracy_score(all_targets, all_preds),
        "precision": precision_score(all_targets, all_preds, zero_division=0),
        "recall": recall_score(all_targets, all_preds, zero_division=0),
        "f1": f1_score(all_targets, all_preds, zero_division=0),
    }


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets, all_probs = [], [], []

    for batch in tqdm(loader, desc="  Val  ", leave=False, ncols=80):
        images = batch["image"].to(device)
        texts = batch["text"].to(device)
        labels = batch["label"].to(device)

        logits = model(texts, images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).cpu().numpy()
        all_preds.extend(preds)
        all_probs.extend(probs.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())

    n = len(loader.dataset)
    return {
        "loss": total_loss / n,
        "accuracy": accuracy_score(all_targets, all_preds),
        "precision": precision_score(all_targets, all_preds, zero_division=0),
        "recall": recall_score(all_targets, all_preds, zero_division=0),
        "f1": f1_score(all_targets, all_preds, zero_division=0),
        "preds": all_preds,
        "probs": all_probs,
        "targets": all_targets,
    }


# ── Main ──────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    if args.config:
        cfg = load_config(args.config)
        for k, v in cfg.items():
            if hasattr(args, k):
                setattr(args, k, v)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = args.run_name or f"{args.fusion}_{args.text_encoder}_{int(time.time())}"

    print(f"╔══════════════════════════════════════════════════════════════╗")
    print(f"║  Multimodal Offensive Meme Detector — Training             ║")
    print(f"╠══════════════════════════════════════════════════════════════╣")
    print(f"║  Device:        {str(device):42s} ║")
    print(f"║  Text encoder:  {args.text_encoder:42s} ║")
    print(f"║  Image backbone:{args.img_backbone:42s} ║")
    print(f"║  Fusion:        {args.fusion:42s} ║")
    print(f"║  Epochs:        {str(args.epochs):42s} ║")
    print(f"║  Batch size:    {str(args.batch_size):42s} ║")
    print(f"║  Learning rate: {str(args.lr):42s} ║")
    print(f"╚══════════════════════════════════════════════════════════════╝")

    # ── Data ───────────────────────────────────────────────────────
    if args.use_mock or not os.path.exists(args.train_csv):
        print("\n⚠  No dataset found — generating mock data...")
        create_mock_dataset("mock_data")
        args.train_csv = "mock_data/train.csv"
        args.val_csv = "mock_data/val.csv"
        args.img_dir = "mock_data/images"

    vocab, embeddings = load_glove_embeddings(args.glove_path, args.glove_dim)

    train_ds = MultiOFFDataset(
        args.train_csv, args.img_dir, vocab,
        transform=TRAIN_AUGMENT if args.augment else None,
    )
    val_ds = MultiOFFDataset(args.val_csv, args.img_dir, vocab)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    # Class distribution
    labels = train_ds.df["label"].values
    n_pos = (labels == 1).sum()
    n_neg = (labels == 0).sum()
    print(f"\n  Train: {len(train_ds)} (offensive={n_pos}, non-offensive={n_neg})")
    print(f"  Val:   {len(val_ds)}")

    # ── Model ──────────────────────────────────────────────────────
    model = MultimodalClassifier(
        embedding_matrix=embeddings,
        text_hidden_dim=args.text_hidden,
        text_encoder=args.text_encoder,
        img_hidden_dim=args.img_hidden,
        img_backbone=args.img_backbone,
        fusion_type=args.fusion,
    ).to(device)

    params = model.count_parameters()
    print(f"\n  Parameters: {params['total']:,} total | "
          f"{params['trainable']:,} trainable | {params['frozen']:,} frozen")

    # ── Optimizer ──────────────────────────────────────────────────
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── Training Loop ──────────────────────────────────────────────
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    best_f1 = 0.0
    patience_counter = 0
    history = []

    hdr = f"{'Ep':>3} │ {'Train Loss':>10} {'Acc':>5} {'P':>5} {'R':>5} {'F1':>5} │ {'Val Loss':>10} {'Acc':>5} {'P':>5} {'R':>5} {'F1':>5}"
    print(f"\n{hdr}")
    print("─" * len(hdr))

    ckpt_path = os.path.join(run_dir, "best_model.pth")
    # Save an initial model just in case it doesn't improve
    torch.save({
        "epoch": 0,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_f1": 0.0,
        "args": vars(args),
        "vocab_size": len(vocab),
        "glove_dim": args.glove_dim,
    }, ckpt_path)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_m = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_m = validate(model, val_loader, criterion, device)
        elapsed = time.time() - t0
        scheduler.step()

        print(
            f"{epoch:3d} │ "
            f"{train_m['loss']:10.4f} {train_m['accuracy']:5.2f} "
            f"{train_m['precision']:5.2f} {train_m['recall']:5.2f} {train_m['f1']:5.2f} │ "
            f"{val_m['loss']:10.4f} {val_m['accuracy']:5.2f} "
            f"{val_m['precision']:5.2f} {val_m['recall']:5.2f} {val_m['f1']:5.2f}  "
            f"({elapsed:.0f}s)"
        )

        history.append({
            "epoch": epoch,
            "train": {k: v for k, v in train_m.items()},
            "val": {k: v for k, v in val_m.items() if k not in ("preds", "probs", "targets")},
        })

        if val_m["f1"] >= best_f1:
            best_f1 = val_m["f1"]
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": best_f1,
                "args": vars(args),
                "vocab_size": len(vocab),
                "glove_dim": args.glove_dim,
            }, ckpt_path)
            print(f"  ✓ Best model saved (F1={best_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n  ⏹ Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                break

    # Save final model + history
    final_path = os.path.join(run_dir, "final_model.pth")
    torch.save(model.state_dict(), final_path)

    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    with open(os.path.join(run_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"\n{'═' * 60}")
    print(f"  Best Val F1: {best_f1:.4f}")
    print(f"  Checkpoint:  {ckpt_path}")
    print(f"  History:     {run_dir}/history.json")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
