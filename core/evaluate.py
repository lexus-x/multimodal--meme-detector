"""
Multimodal Meme Offensive Content Detection — Evaluation

Usage:
    python -m core.evaluate --checkpoint checkpoints/cross_attention_bilstm/best_model.pth
    python -m core.evaluate --checkpoint checkpoints/early_bilstm/best_model.pth --use_mock
"""

import os
import argparse
import json

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, average_precision_score,
)
from tqdm import tqdm

from core.dataset import MultiOFFDataset, load_glove_embeddings, create_mock_dataset
from core.models import MultimodalClassifier


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate offensive meme detector")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    p.add_argument("--test_csv", type=str, default="data/processed/test.csv")
    p.add_argument("--img_dir", type=str, default="data/processed/images")
    p.add_argument("--glove_path", type=str, default="glove.6B/glove.6B.50d.txt")
    p.add_argument("--glove_dim", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--use_mock", action="store_true")
    p.add_argument("--output_dir", type=str, default="outputs")
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_probs, all_targets, all_sentences = [], [], [], []

    for batch in tqdm(loader, desc="Evaluating", ncols=80):
        images = batch["image"].to(device)
        texts = batch["text"].to(device)
        labels = batch["label"]

        logits = model(texts, images)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs > 0.5).astype(int)

        all_probs.extend(probs)
        all_preds.extend(preds)
        all_targets.extend(labels.numpy())
        all_sentences.extend(batch.get("sentence", [""] * len(labels)))

    return {
        "targets": np.array(all_targets),
        "preds": np.array(all_preds),
        "probs": np.array(all_probs),
        "sentences": all_sentences,
    }


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})

    # Reconstruct model
    text_encoder = ckpt_args.get("text_encoder", "bilstm")
    text_hidden = ckpt_args.get("text_hidden", 128)
    img_hidden = ckpt_args.get("img_hidden", 256)
    img_backbone = ckpt_args.get("img_backbone", "vgg16")
    fusion_type = ckpt_args.get("fusion", "early")
    glove_dim = ckpt_args.get("glove_dim", args.glove_dim)

    # Load vocab
    if args.use_mock or not os.path.exists(args.test_csv):
        create_mock_dataset("mock_data")
        args.test_csv = "mock_data/test.csv"
        args.img_dir = "mock_data/images"

    vocab, embeddings = load_glove_embeddings(args.glove_path, glove_dim)

    model = MultimodalClassifier(
        embedding_matrix=embeddings,
        text_hidden_dim=text_hidden,
        text_encoder=text_encoder,
        img_hidden_dim=img_hidden,
        img_backbone=img_backbone,
        fusion_type=fusion_type,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')} (val_f1={ckpt.get('val_f1', '?')})")

    # Test dataset
    test_ds = MultiOFFDataset(args.test_csv, args.img_dir, vocab)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Evaluate
    results = evaluate(model, test_loader, device)
    targets = results["targets"]
    preds = results["preds"]
    probs = results["probs"]

    # ── Metrics ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  TEST RESULTS")
    print("=" * 60)
    print(classification_report(
        targets, preds,
        target_names=["Non-offensive", "Offensive"],
        digits=4, zero_division=0,
    ))

    cm = confusion_matrix(targets, preds)
    print("Confusion Matrix:")
    print(f"  TN={cm[0][0]:4d}  FP={cm[0][1]:4d}")
    print(f"  FN={cm[1][0]:4d}  TP={cm[1][1]:4d}")

    try:
        auc = roc_auc_score(targets, probs)
        ap = average_precision_score(targets, probs)
        print(f"\n  ROC-AUC:     {auc:.4f}")
        print(f"  Avg Prec:    {ap:.4f}")
    except ValueError:
        auc, ap = 0.0, 0.0
        print("\n  ROC-AUC:     N/A (single class in targets)")

    # ── Error Analysis ─────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  ERROR ANALYSIS")
    print("─" * 60)

    fp_mask = (preds == 1) & (targets == 0)
    fn_mask = (preds == 0) & (targets == 1)

    if fp_mask.any():
        print(f"\n  False Positives ({fp_mask.sum()} cases) — "
              "non-offensive memes flagged as offensive:")
        fp_indices = np.where(fp_mask)[0][:5]
        for i in fp_indices:
            print(f"    [{probs[i]:.2f}] {results['sentences'][i] if i < len(results['sentences']) else 'N/A'}")

    if fn_mask.any():
        print(f"\n  False Negatives ({fn_mask.sum()} cases) — "
              "offensive memes that slipped through:")
        fn_indices = np.where(fn_mask)[0][:5]
        for i in fn_indices:
            print(f"    [{probs[i]:.2f}] {results['sentences'][i] if i < len(results['sentences']) else 'N/A'}")

    # ── Save ───────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "eval_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "checkpoint": args.checkpoint,
            "fusion": fusion_type,
            "text_encoder": text_encoder,
            "confusion_matrix": cm.tolist(),
            "roc_auc": float(auc),
            "avg_precision": float(ap),
            "classification_report": classification_report(
                targets, preds, target_names=["Non-offensive", "Offensive"],
                digits=4, zero_division=0, output_dict=True,
            ),
        }, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
