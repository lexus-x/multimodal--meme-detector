#!/usr/bin/env python3
"""Evaluate all checkpoints and produce comparison table."""
import os
import json
import torch
import argparse
import glob
from core.models import MultimodalClassifier
from core.dataset import MultiOFFDataset, load_glove_embeddings, create_mock_dataset
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints_dir", default="research/checkpoints")
    parser.add_argument("--test_csv", default="data/processed/test.csv")
    parser.add_argument("--img_dir", default="data/processed/images")
    parser.add_argument("--glove_path", default="research/data/glove.6B/glove.6B.50d.txt")
    parser.add_argument("--use_mock", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.use_mock or not os.path.exists(args.test_csv):
        create_mock_dataset("mock_data")
        args.test_csv = "mock_data/test.csv"
        args.img_dir = "mock_data/images"

    vocab, embeddings = load_glove_embeddings(args.glove_path, 50)
    test_ds = MultiOFFDataset(args.test_csv, args.img_dir, vocab)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    # Find all checkpoints
    ckpts = glob.glob(os.path.join(args.checkpoints_dir, "*/best_model.pth"))
    if not ckpts:
        print("No checkpoints found!")
        return

    results = []
    for ckpt_path in sorted(ckpts):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        ckpt_args = ckpt.get("args", {})

        model = MultimodalClassifier(
            embedding_matrix=embeddings,
            text_hidden_dim=ckpt_args.get("text_hidden", 128),
            text_encoder=ckpt_args.get("text_encoder", "bilstm"),
            img_hidden_dim=ckpt_args.get("img_hidden", 256),
            img_backbone=ckpt_args.get("img_backbone", "vgg16"),
            fusion_type=ckpt_args.get("fusion", "early"),
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        all_probs, all_targets = [], []
        with torch.no_grad():
            for batch in test_loader:
                logits = model(batch["text"].to(device), batch["image"].to(device))
                all_probs.extend(torch.sigmoid(logits).cpu().numpy())
                all_targets.extend(batch["label"].numpy())

        import numpy as np
        probs = np.array(all_probs)
        targets = np.array(all_targets)
        preds = (probs > 0.5).astype(int)

        name = os.path.basename(os.path.dirname(ckpt_path))
        results.append({
            "name": name,
            "fusion": ckpt_args.get("fusion", "?"),
            "text_encoder": ckpt_args.get("text_encoder", "?"),
            "f1": f1_score(targets, preds, zero_division=0),
            "precision": precision_score(targets, preds, zero_division=0),
            "recall": recall_score(targets, preds, zero_division=0),
            "auc": roc_auc_score(targets, probs) if len(set(targets)) > 1 else 0,
        })

    # Print table
    print(f"\n{'─' * 85}")
    print(f"{'Model':<30} {'Fusion':<18} {'Encoder':<10} {'F1':>6} {'P':>6} {'R':>6} {'AUC':>6}")
    print(f"{'─' * 85}")
    for r in sorted(results, key=lambda x: x["f1"], reverse=True):
        print(f"{r['name']:<30} {r['fusion']:<18} {r['text_encoder']:<10} "
              f"{r['f1']:6.3f} {r['precision']:6.3f} {r['recall']:6.3f} {r['auc']:6.3f}")
    print(f"{'─' * 85}")


if __name__ == "__main__":
    main()
