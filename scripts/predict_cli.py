#!/usr/bin/env python3
"""CLI tool for single meme prediction."""
import argparse
import torch
from PIL import Image
from torchvision import transforms
from core.dataset import load_glove_embeddings
from core.models import MultimodalClassifier


def main():
    parser = argparse.ArgumentParser(description="Predict offensive content in a meme")
    parser.add_argument("--image", required=True, help="Path to meme image")
    parser.add_argument("--text", required=True, help="Meme text content")
    parser.add_argument("--checkpoint", default="research/checkpoints/best_run/best_model.pth")
    parser.add_argument("--glove_path", default="research/data/glove.6B/glove.6B.50d.txt")
    parser.add_argument("--fusion", default=None, help="Override fusion type")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})

    # Load embeddings
    vocab, embeddings = load_glove_embeddings(args.glove_path, ckpt_args.get("glove_dim", 50))

    # Build model
    model = MultimodalClassifier(
        embedding_matrix=embeddings,
        text_hidden_dim=ckpt_args.get("text_hidden", 128),
        text_encoder=ckpt_args.get("text_encoder", "bilstm"),
        img_hidden_dim=ckpt_args.get("img_hidden", 256),
        img_backbone=ckpt_args.get("img_backbone", "vgg16"),
        fusion_type=args.fusion or ckpt_args.get("fusion", "cross_attention"),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Preprocess
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = transform(Image.open(args.image).convert("RGB")).unsqueeze(0).to(device)
    tokens = args.text.lower().split()
    indices = [vocab.get(t, vocab.get("<unk>", 1)) for t in tokens]
    if len(indices) < 50:
        indices += [0] * (50 - len(indices))
    else:
        indices = indices[:50]
    text = torch.tensor([indices], dtype=torch.long).to(device)

    # Predict
    with torch.no_grad():
        prob = torch.sigmoid(model(text, image)).item()

    label = "🚫 OFFENSIVE" if prob > 0.5 else "✅ NON-OFFENSIVE"
    print(f"\n{label}")
    print(f"  Offensive probability: {prob:.1%}")
    print(f"  Confidence: {max(prob, 1-prob):.1%}")
    print(f"  Model: {args.checkpoint}")
    print(f"  Fusion: {args.fusion or ckpt_args.get('fusion', 'unknown')}")


if __name__ == "__main__":
    main()
