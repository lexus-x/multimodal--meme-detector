"""
Multimodal Meme Offensive Content Detection — Live Experiment UI

A Gradio-based web app for interactive experimentation.
Upload a meme image + enter text → get offensive/non-offensive prediction
with confidence scores, and optionally compare multiple fusion strategies.

Usage:
    python -m core.demo
    python -m core.demo --checkpoint checkpoints/cross_attention_bilstm/best_model.pth
    python -m core.demo --share  (creates public link)
"""

import os
import argparse
import json
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from core.dataset import load_glove_embeddings
from core.models import MultimodalClassifier
from core.explain import get_attention_weights, generate_explanation_text, explain_prediction

# ── Globals ───────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS: dict[str, MultimodalClassifier] = {}
VOCAB: dict = None
EMBEDDINGS: torch.Tensor = None
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

FUSION_LABELS = {
    "early": "Early Fusion (Baseline)",
    "cross_attention": "Cross-Modal Attention",
    "cross_attention_deep": "Deep Cross-Modal Attention",
    "gated": "Gated Fusion (Neverova 2016)",
    "bilinear": "Bilinear Fusion (Tsai 2017)",
}

MAX_LEN = 50


# ── Model Loading ─────────────────────────────────────────────────────

def load_model(checkpoint_path: str) -> MultimodalClassifier:
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    args = ckpt.get("args", {})

    model = MultimodalClassifier(
        embedding_matrix=EMBEDDINGS,
        text_hidden_dim=args.get("text_hidden", 128),
        text_encoder=args.get("text_encoder", "bilstm"),
        img_hidden_dim=args.get("img_hidden", 256),
        img_backbone=args.get("img_backbone", "vgg16"),
        fusion_type=args.get("fusion", "early"),
    ).to(DEVICE)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def load_all_models(checkpoints_dir: str):
    """Auto-discover and load all trained models from checkpoints/."""
    if not os.path.exists(checkpoints_dir):
        print(f"  ⚠ No checkpoints dir found at {checkpoints_dir}")
        return

    for entry in os.listdir(checkpoints_dir):
        run_dir = os.path.join(checkpoints_dir, entry)
        best = os.path.join(run_dir, "best_model.pth")
        if os.path.isfile(best):
            try:
                model = load_model(best)
                name = entry  # e.g. "cross_attention_bilstm_1234567"
                # Clean up name
                parts = name.rsplit("_", 1)
                label = parts[0] if len(parts) > 1 and parts[1].isdigit() else name
                MODELS[label] = model
                print(f"  ✓ Loaded model: {label}")
            except Exception as e:
                print(f"  ✗ Failed to load {entry}: {e}")


# ── Inference ─────────────────────────────────────────────────────────

def tokenize(text: str) -> torch.Tensor:
    tokens = str(text).lower().split()
    indices = [VOCAB.get(t, VOCAB.get("<unk>", 1)) for t in tokens]
    if len(indices) < MAX_LEN:
        indices += [0] * (MAX_LEN - len(indices))
    else:
        indices = indices[:MAX_LEN]
    return torch.tensor(indices, dtype=torch.long)


@torch.no_grad()
def predict_single(model: MultimodalClassifier, image: Image.Image, text: str) -> dict:
    img_tensor = TRANSFORM(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    txt_tensor = tokenize(text).unsqueeze(0).to(DEVICE)

    logits = model(txt_tensor, img_tensor)
    prob = torch.sigmoid(logits).item()

    return {
        "offensive_prob": prob,
        "non_offensive_prob": 1 - prob,
        "prediction": "Offensive" if prob > 0.5 else "Non-offensive",
        "confidence": max(prob, 1 - prob),
    }


# ── Gradio Handlers ───────────────────────────────────────────────────

def classify_meme(image: Image.Image, text: str, model_name: str) -> tuple:
    """Main classification handler."""
    if image is None:
        return None, "⚠ Please upload an image", {}

    if not text.strip():
        return None, "⚠ Please enter the meme text", {}

    if model_name not in MODELS:
        return None, f"⚠ Model '{model_name}' not found", {}

    model = MODELS[model_name]
    result = predict_single(model, image, text)

    # Build result HTML
    label = result["prediction"]
    conf = result["confidence"]
    prob_off = result["offensive_prob"]
    prob_non = result["non_offensive_prob"]

    color = "#e74c3c" if label == "Offensive" else "#27ae60"
    emoji = "🚫" if label == "Offensive" else "✅"

    html = f"""
    <div style="padding: 20px; border-radius: 12px; background: {color}15; border: 2px solid {color};">
        <h2 style="color: {color}; margin: 0;">{emoji} {label}</h2>
        <p style="font-size: 1.2em; margin: 8px 0;">
            <b>Confidence:</b> {conf:.1%}
        </p>
        <div style="margin: 12px 0;">
            <div style="display: flex; align-items: center; margin: 6px 0;">
                <span style="width: 140px;">Non-offensive:</span>
                <div style="flex: 1; background: #eee; border-radius: 4px; height: 20px;">
                    <div style="width: {prob_non*100:.1f}%; background: #27ae60; height: 100%; border-radius: 4px;"></div>
                </div>
                <span style="width: 60px; text-align: right;">{prob_non:.1%}</span>
            </div>
            <div style="display: flex; align-items: center; margin: 6px 0;">
                <span style="width: 140px;">Offensive:</span>
                <div style="flex: 1; background: #eee; border-radius: 4px; height: 20px;">
                    <div style="width: {prob_off*100:.1f}%; background: #e74c3c; height: 100%; border-radius: 4px;"></div>
                </div>
                <span style="width: 60px; text-align: right;">{prob_off:.1%}</span>
            </div>
        </div>
        <p style="font-size: 0.9em; color: #666; margin-top: 8px;">
            Model: <code>{model_name}</code>
        </p>
    </div>
    """
    return html, label, {"offensive": prob_off, "non_offensive": prob_non}


def compare_models(image: Image.Image, text: str) -> str:
    """Compare all loaded models on the same input."""
    if image is None or not text.strip():
        return "⚠ Upload an image and enter text to compare models"

    results = []
    for name, model in sorted(MODELS.items()):
        r = predict_single(model, image, text)
        results.append((name, r))

    # Build comparison HTML
    rows = ""
    for name, r in results:
        label = r["prediction"]
        color = "#e74c3c" if label == "Offensive" else "#27ae60"
        emoji = "🚫" if label == "Offensive" else "✅"
        rows += f"""
        <tr>
            <td style="padding: 8px; border-bottom: 1px solid #eee;"><code>{name}</code></td>
            <td style="padding: 8px; border-bottom: 1px solid #eee; color: {color}; font-weight: bold;">
                {emoji} {label}
            </td>
            <td style="padding: 8px; border-bottom: 1px solid #eee;">{r['confidence']:.1%}</td>
            <td style="padding: 8px; border-bottom: 1px solid #eee;">{r['offensive_prob']:.3f}</td>
        </tr>
        """

    html = f"""
    <div style="padding: 10px;">
        <h3>🔬 Model Comparison</h3>
        <table style="width: 100%; border-collapse: collapse;">
            <tr style="background: #f8f9fa;">
                <th style="padding: 10px; text-align: left;">Model</th>
                <th style="padding: 10px; text-align: left;">Prediction</th>
                <th style="padding: 10px; text-align: left;">Confidence</th>
                <th style="padding: 10px; text-align: left;">P(Offensive)</th>
            </tr>
            {rows}
        </table>
    </div>
    """
    return html


# ── Example Cases ─────────────────────────────────────────────────────

EXAMPLES = [
    ["examples/meme1.jpg", "When you finally fix the bug but create 5 more"],
    ["examples/meme2.jpg", "You are absolutely terrible at this"],
    ["examples/meme3.jpg", "Me pretending to work while the boss walks by"],
    [None, "I love spending time with my family on weekends"],
    [None, "Go away nobody wants you here you idiot"],
]


# ── Build UI ──────────────────────────────────────────────────────────

def build_demo():
    import gradio as gr

    model_names = list(MODELS.keys())
    default_model = model_names[0] if model_names else None

    # ── Theme ──────────────────────────────────────────────────────
    theme = gr.themes.Soft(
        primary_hue="blue",
        neutral_hue="slate",
    )

    with gr.Blocks(
        title="🔍 Offensive Meme Detector",
        theme=theme,
        css="""
        .main-title { text-align: center; margin-bottom: 0; }
        .subtitle { text-align: center; color: #666; margin-top: 4px; }
        """,
    ) as demo:
        gr.HTML("""
        <div style="padding: 20px 0;">
            <h1 class="main-title">🔍 Multimodal Offensive Meme Detector</h1>
            <p class="subtitle">
                Multimodal deep learning for detecting offensive content in memes<br>
                <span style="font-size: 0.85em; color: #999;">
                    Text (GloVe + BiLSTM/CNN) × Image (VGG16/ResNet) × Fusion (Early / Cross-Attention / Gated / Bilinear)
                </span>
            </p>
        </div>
        """)

        with gr.Tabs():
            # ── Tab 1: Classify ────────────────────────────────────
            with gr.Tab("🎯 Classify", id="classify"):
                with gr.Row():
                    with gr.Column(scale=1):
                        image_input = gr.Image(type="pil", label="Upload Meme Image")
                        text_input = gr.Textbox(
                            label="Meme Text",
                            placeholder="Enter the text from the meme...",
                            lines=2,
                        )
                        model_dropdown = gr.Dropdown(
                            choices=model_names,
                            value=default_model,
                            label="Select Model",
                        )
                        classify_btn = gr.Button("🔍 Classify", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        result_html = gr.HTML(label="Result")
                        result_label = gr.Textbox(visible=False)
                        result_scores = gr.JSON(visible=False)

                classify_btn.click(
                    fn=classify_meme,
                    inputs=[image_input, text_input, model_dropdown],
                    outputs=[result_html, result_label, result_scores],
                )

            # ── Tab 2: Compare ─────────────────────────────────────
            with gr.Tab("🔬 Compare Models", id="compare"):
                gr.Markdown("### Run all models on the same input and compare results")
                with gr.Row():
                    with gr.Column(scale=1):
                        cmp_image = gr.Image(type="pil", label="Upload Meme Image")
                        cmp_text = gr.Textbox(label="Meme Text", placeholder="Enter meme text...", lines=2)
                        compare_btn = gr.Button("🔬 Compare All Models", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        cmp_result = gr.HTML(label="Comparison Results")

                compare_btn.click(
                    fn=compare_models,
                    inputs=[cmp_image, cmp_text],
                    outputs=[cmp_result],
                )

            # ── Tab 3: Architecture ────────────────────────────────
            with gr.Tab("🏗️ Architecture", id="arch"):
                gr.Markdown("""
                ### Model Architecture

                A multimodal classification system using pretrained backbones
                and multiple fusion strategies. This is a **lightweight baseline**
                — not SOTA. See "Where We Stand" below.

                ```
                ┌─────────────────────────┐    ┌─────────────────────────┐
                │      TEXT BRANCH         │    │     IMAGE BRANCH        │
                │                          │    │                         │
                │  GloVe Embeddings (50d)  │    │  VGG16 / ResNet50       │
                │         ↓                │    │  (pretrained, frozen)   │
                │  BiLSTM / LSTM / CNN     │    │         ↓               │
                │         ↓                │    │  FC projection          │
                │  text_feat (256d)        │    │  img_feat (256d)        │
                └───────────┬──────────────┘    └───────────┬─────────────┘
                            │                               │
                            └───────────┬───────────────────┘
                                        ↓
                              ┌──────────────────┐
                              │   FUSION LAYER    │
                              │                   │
                              │  • Early (concat) │  ← Baseline
                              │  • Cross-Attn     │  ← Attention-based
                              │  • Gated          │  ← Modality weighting
                              │  • Bilinear       │  ← Second-order
                              └────────┬─────────┘
                                       ↓
                              ┌──────────────────┐
                              │  MLP CLASSIFIER   │
                              │  512→256→128→1    │
                              └──────────────────┘
                               Output: P(offensive)
                ```

                ### Fusion Strategies

                | Fusion | Description | Reference |
                |--------|-------------|-----------|
                | **Early** | Simple concatenation of features | Baseline |
                | **Cross-Attention** | Multi-head attention across modalities | Standard (ViLBERT-style) |
                | **Gated** | Learned gate weights each modality | Neverova et al., 2016 |
                | **Bilinear** | Second-order feature interactions | Tsai et al., 2017 |

                ### Where We Stand

                **This project uses classic multimodal architectures** (BiLSTM + VGG16
                with various fusion). It is NOT SOTA.

                **Actual SOTA** on meme hate/offensive detection uses large multimodal
                transformers:

                | Model | Dataset | AUROC | Source |
                |-------|---------|-------|--------|
                | RA-HMD (LMM fine-tuning) | Hateful Memes | ~87.0 | EMNLP 2025 |
                | Retrieval-guided LMM | Hateful Memes | 87.0 | arXiv 2024 |
                | CLIP + ResNet50 | Hateful Memes | 81.7 | AICS 2024 |
                | VisualBERT | Hateful Memes | ~82 | Kiela et al., 2020 |
                | UNITER | Hateful Memes | ~83 | Kiela et al., 2020 |
                | **BiLSTM + VGG16 (ours)** | MultiOFF | ~65 F1 | **This project** |
                | BiLSTM + VGG16 (original) | MultiOFF | ~41 F1 | Alam et al., 2020 |

                Our cross-modal attention fusion improves over the original MultiOFF
                baselines (F1 ~41 → ~65), but is far from SOTA on larger benchmarks.
                The value of this project is as an **interpretable, lightweight baseline**
                that works on small datasets without massive pretrained models.

                ### Datasets in This Domain

                | Dataset | Samples | Task | Notes |
                |---------|---------|------|-------|
                | **MultiOFF** | 743 | Binary offensive | Our primary dataset |
                | Hateful Memes | 10,000 | Binary hateful | Facebook AI, gold standard |
                | Memotion 1.0 | 8,898 | Multi-class sentiment | Humor, sarcasm, offensive |
                | Harm-C | 3,035 | Harmful content | Fine-grained harm types |
                | MAMI | 12,000 | Misogynous memes | Multimedia misogyny |
                """)

            # ── Tab 4: About ───────────────────────────────────────
            with gr.Tab("📖 About", id="about"):
                gr.Markdown("""
                ### Multimodal Meme Offensive Content Detection

                **Problem:** Memes combine images and text to convey meaning.
                Offensive intent often emerges only when both modalities are
                interpreted together — making single-modal detection insufficient.

                **Approach:** We implement and compare multiple multimodal fusion
                strategies on top of pretrained backbones:
                - **Text:** GloVe embeddings + BiLSTM/LSTM/CNN encoders
                - **Image:** VGG16/ResNet50 pretrained on ImageNet (frozen)
                - **Fusion:** Early concat, cross-modal attention, gated, bilinear

                **What this project is:**
                - A reproducible comparison of fusion strategies on a small meme dataset
                - An educational codebase with clean, modular architecture
                - A live experiment UI for interactive testing
                - An improvement over the original MultiOFF baselines (F1 ~41 → ~65)

                **What this project is NOT:**
                - SOTA on hateful meme detection (that requires large multimodal transformers)
                - Trained on a large dataset (MultiOFF has only 743 samples)
                - A replacement for production content moderation systems

                **Key references:**
                - Alam et al., "MultiOFF Dataset" (LREC 2020) — our dataset
                - Kiela et al., "Hateful Memes Challenge" (NeurIPS 2020) — the gold standard benchmark
                - Neverova et al., "ModDrop" (TPAMI 2016) — gated fusion
                - Tsai et al., "Multimodal Fusion" (ACM MM 2017) — bilinear fusion
                - Lu et al., "ViLBERT" (NeurIPS 2019) — cross-modal attention (reference)

                ---

                *Built with PyTorch • Gradio • scikit-learn*
                """)

            # ── Tab 5: Explain ─────────────────────────────────────
            with gr.Tab("🧠 Explain", id="explain"):
                gr.Markdown("### Model Explainability — Understand why the model made its prediction")
                with gr.Row():
                    with gr.Column(scale=1):
                        exp_image = gr.Image(type="pil", label="Upload Meme Image")
                        exp_text = gr.Textbox(label="Meme Text", placeholder="Enter meme text...", lines=2)
                        exp_model = gr.Dropdown(
                            choices=model_names, value=default_model,
                            label="Select Model",
                        )
                        explain_btn = gr.Button("🧠 Explain Prediction", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        exp_result = gr.Markdown(label="Explanation")

                def explain_meme(image, text, model_name):
                    if image is None or not text.strip():
                        return "⚠ Please upload an image and enter text"
                    if model_name not in MODELS:
                        return f"⚠ Model '{model_name}' not found"

                    model = MODELS[model_name]
                    img_tensor = TRANSFORM(image.convert("RGB")).unsqueeze(0).to(DEVICE)
                    txt_tensor = tokenize(text).unsqueeze(0).to(DEVICE)

                    explanation = explain_prediction(model, txt_tensor, img_tensor, VOCAB)
                    return generate_explanation_text(explanation)

                explain_btn.click(
                    fn=explain_meme,
                    inputs=[exp_image, exp_text, exp_model],
                    outputs=[exp_result],
                )

    return demo


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Launch Gradio demo")
    parser.add_argument("--checkpoints", type=str, default="checkpoints")
    parser.add_argument("--glove_path", type=str, default="glove.6B/glove.6B.50d.txt")
    parser.add_argument("--glove_dim", type=int, default=50)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    args = parser.parse_args()

    global VOCAB, EMBEDDINGS

    print("╔══════════════════════════════════════════════════════════╗")
    print("║  Offensive Meme Detector — Live Experiment UI          ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"\n  Device: {DEVICE}")

    # Load embeddings
    VOCAB, EMBEDDINGS = load_glove_embeddings(args.glove_path, args.glove_dim)

    # Load all models
    print(f"\n  Loading models from {args.checkpoints}/...")
    load_all_models(args.checkpoints)

    if not MODELS:
        print("\n  ⚠ No trained models found. Running in demo mode with random weights.")
        # Create a dummy model for demo
        model = MultimodalClassifier(
            embedding_matrix=EMBEDDINGS,
            text_encoder="bilstm",
            fusion_type="cross_attention",
        ).to(DEVICE)
        model.eval()
        MODELS["demo_cross_attention"] = model
        MODELS["demo_early"] = MultimodalClassifier(
            embedding_matrix=EMBEDDINGS, fusion_type="early"
        ).to(DEVICE)
        MODELS["demo_early"].eval()

    print(f"\n  Models loaded: {list(MODELS.keys())}")

    # Launch
    demo = build_demo()
    print(f"\n  Launching on port {args.port}...")
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
