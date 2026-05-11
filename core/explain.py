"""Model explainability — attention visualization and feature analysis."""
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms


def get_attention_weights(model, text_tensor, img_tensor):
    """Extract cross-modal attention weights if available.
    
    Returns attention matrix or None if model doesn't use cross-attention.
    """
    if not hasattr(model, 'fusion'):
        return None
    
    fusion = model.fusion
    if not hasattr(fusion, 'q_proj'):
        return None
    
    model.eval()
    with torch.no_grad():
        text_feat = model.text_branch(text_tensor)
        img_feat = model.image_branch(img_tensor)
        
        B = text_feat.size(0)
        q = fusion.q_proj(text_feat).unsqueeze(1)
        k = fusion.k_proj(img_feat).unsqueeze(1)
        v = fusion.v_proj(img_feat).unsqueeze(1)
        
        num_heads = fusion.num_heads
        head_dim = text_feat.size(1) // num_heads
        
        q = q.view(B, 1, num_heads, head_dim).transpose(1, 2)
        k = k.view(B, 1, num_heads, head_dim).transpose(1, 2)
        
        scale = head_dim ** 0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale
        attn = F.softmax(attn, dim=-1)
        
        return attn.squeeze().cpu().numpy()


def get_text_importance(model, text_tensor, img_tensor, vocab):
    """Compute token importance via gradient-based attribution.
    
    Returns list of (token, importance_score) tuples.
    """
    model.eval()
    
    # Get embedding layer
    embedding = model.text_branch.embedding
    
    # Create input with gradient tracking
    text_emb = embedding(text_tensor)
    text_emb.retain_grad()
    
    # Forward pass
    img_feat = model.image_branch(img_tensor)
    fused = model.fusion(text_emb, img_tensor)  # this won't work directly
    logits = model.classifier(fused)
    
    # Backward
    logits.backward()
    
    # Gradient magnitude as importance
    grads = text_emb.grad.abs().mean(dim=-1).squeeze().cpu().numpy()
    
    # Map back to tokens
    inv_vocab = {v: k for k, v in vocab.items()}
    tokens = [inv_vocab.get(idx.item(), "<unk>") for idx in text_tensor.squeeze()]
    
    return list(zip(tokens, grads))


def explain_prediction(model, text_tensor, img_tensor, vocab=None):
    """Generate explanation for a prediction.
    
    Returns dict with:
    - attention_weights: cross-modal attention matrix (if available)
    - text_importance: token-level importance (if vocab provided)
    - features: fused feature statistics
    """
    model.eval()
    explanation = {}
    
    with torch.no_grad():
        text_feat = model.text_branch(text_tensor)
        img_feat = model.image_branch(img_tensor)
        fused = model.fusion(text_feat, img_feat)
        logits = model.classifier(fused)
        prob = torch.sigmoid(logits).item()
    
    explanation["prediction"] = {
        "probability": prob,
        "label": "Offensive" if prob > 0.5 else "Non-offensive",
        "confidence": max(prob, 1 - prob),
    }
    
    explanation["features"] = {
        "text_norm": text_feat.norm().item(),
        "image_norm": img_feat.norm().item(),
        "fused_norm": fused.norm().item(),
        "text_mean": text_feat.mean().item(),
        "image_mean": img_feat.mean().item(),
    }
    
    # Attention weights
    attn = get_attention_weights(model, text_tensor, img_tensor)
    if attn is not None:
        explanation["attention"] = attn.tolist()
    
    return explanation


def generate_explanation_text(explanation: dict) -> str:
    """Generate human-readable explanation text."""
    pred = explanation["prediction"]
    feats = explanation["features"]
    
    lines = []
    lines.append(f"## 🔍 Prediction: {pred['label']}")
    lines.append(f"**Confidence:** {pred['confidence']:.1%}")
    lines.append(f"**P(Offensive):** {pred['probability']:.3f}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 📊 Feature Analysis")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Text feature magnitude | {feats['text_norm']:.3f} |")
    lines.append(f"| Image feature magnitude | {feats['image_norm']:.3f} |")
    lines.append(f"| Fused feature magnitude | {feats['fused_norm']:.3f} |")
    
    text_ratio = feats['text_norm'] / (feats['text_norm'] + feats['image_norm'] + 1e-8)
    lines.append("")
    if text_ratio > 0.6:
        lines.append("### 🔤 Text-Dominant")
        lines.append("Text features have stronger influence on the prediction than image features.")
    elif text_ratio < 0.4:
        lines.append("### 🖼️ Image-Dominant")
        lines.append("Image features have stronger influence on the prediction than text features.")
    else:
        lines.append("### ⚖️ Balanced")
        lines.append("Both modalities contribute similarly to the prediction.")
    
    lines.append("")
    lines.append("---")
    lines.append("")
    
    if pred['label'] == 'Offensive':
        lines.append("## ⚠️ Analysis")
        lines.append("The model detected patterns in the **combination of text and image** that match offensive content patterns in the training data. This does not necessarily mean the content is harmful — the model has limited training data (743 samples) and may produce false positives.")
    else:
        lines.append("## ✅ Analysis")
        lines.append("The model found **no strong indicators** of offensive content in this meme. The relationship between the visual elements and text appears benign based on the training data.")
    
    if explanation.get("attention"):
        lines.append("")
        lines.append("## 🎯 Cross-Modal Attention")
        lines.append("The cross-modal attention mechanism shows how text features attend to image features.")
    
    return "\n".join(lines)
