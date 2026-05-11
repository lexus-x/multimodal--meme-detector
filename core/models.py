"""
Multimodal Meme Offensive Content Detection — Models

Multiple fusion strategies for multimodal classification:
  1. Early Fusion           — concat features (baseline)
  2. Cross-Modal Attention  — multi-head attention across modalities
  3. Gated Fusion           — learned modality weighting (Neverova et al., 2016)
  4. Bilinear Fusion        — second-order interactions (Tsai et al., 2017)

All share the same TextBranch / ImageBranch backbones.

NOTE: This is a research baseline, not SOTA. Current SOTA on hateful meme
detection uses large multimodal transformers (VisualBERT, UNITER, CLIP-based
models) achieving AUROC >87 on Hateful Memes benchmark. Our approach uses
lightweight backbones (BiLSTM + VGG16) suitable for small datasets and
low-resource settings.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ═══════════════════════════════════════════════════════════════════════
#  Shared Backbone Components
# ═══════════════════════════════════════════════════════════════════════

class TextBranch(nn.Module):
    """Encodes tokenized text into a fixed-size feature vector.

    Supports: lstm, bilstm, cnn (multi-kernel 1D CNN)
    """
    SUPPORTED = ("lstm", "bilstm", "cnn")

    def __init__(self, embedding_matrix: torch.Tensor, hidden_dim: int = 128,
                 encoder_type: str = "bilstm", dropout: float = 0.3):
        super().__init__()
        vocab_size, emb_dim = embedding_matrix.shape
        encoder_type = encoder_type.lower()
        if encoder_type not in self.SUPPORTED:
            raise ValueError(f"encoder_type must be one of {self.SUPPORTED}")

        self.encoder_type = encoder_type
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix, freeze=False, padding_idx=0
        )

        if encoder_type in ("lstm", "bilstm"):
            self.rnn = nn.LSTM(
                emb_dim, hidden_dim, batch_first=True,
                bidirectional=(encoder_type == "bilstm"),
                dropout=dropout if encoder_type == "lstm" else 0,
            )
            self.output_dim = hidden_dim * (2 if encoder_type == "bilstm" else 1)

        elif encoder_type == "cnn":
            self.convs = nn.ModuleList([
                nn.Conv1d(emb_dim, hidden_dim, kernel_size=k, padding=k // 2)
                for k in (3, 4, 5)
            ])
            self.relu = nn.ReLU()
            self.output_dim = hidden_dim * 3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)

        if self.encoder_type in ("lstm", "bilstm"):
            _, (hidden, _) = self.rnn(embedded)
            if self.encoder_type == "bilstm":
                return torch.cat((hidden[-2], hidden[-1]), dim=1)
            return hidden[-1]

        elif self.encoder_type == "cnn":
            embedded = embedded.permute(0, 2, 1)
            pooled = [torch.max(self.relu(conv(embedded)), dim=2)[0] for conv in self.convs]
            return torch.cat(pooled, dim=1)


class ImageBranch(nn.Module):
    """Extracts visual features using VGG16 with optional fine-tuning."""
    def __init__(self, hidden_dim: int = 256, freeze_backbone: bool = True):
        super().__init__()
        vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        if freeze_backbone:
            for param in vgg16.features.parameters():
                param.requires_grad = False

        self.features = vgg16.features
        self.avgpool = vgg16.avgpool
        self.fc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class ImageBranchResNet(nn.Module):
    """Alternative image backbone using ResNet50 (stronger features)."""
    def __init__(self, hidden_dim: int = 256, freeze_backbone: bool = True):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        if freeze_backbone:
            for param in resnet.layer4.parameters():
                param.requires_grad = False

        # Remove the final FC, keep everything up to avgpool
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # (batch, 2048, 1, 1)
        self.fc = nn.Sequential(
            nn.Linear(2048, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# ═══════════════════════════════════════════════════════════════════════
#  Fusion Strategies
# ═══════════════════════════════════════════════════════════════════════

class EarlyFusion(nn.Module):
    """Simple concatenation — baseline approach."""
    def __init__(self, text_dim: int, img_dim: int):
        super().__init__()
        self.output_dim = text_dim + img_dim

    def forward(self, text_feat: torch.Tensor, img_feat: torch.Tensor) -> torch.Tensor:
        return torch.cat((text_feat, img_feat), dim=1)


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention: text queries attend to image keys/values.

    This is a standard multi-head attention mechanism applied across
    modalities (as used in ViLBERT, UNITER, etc.). For this project,
    we apply it as a fusion strategy on top of BiLSTM+VGG16 features,
    which improves over simple concatenation on small datasets like
    MultiOFF where full multimodal transformers would overfit.

    Architecture:
      Q = W_q @ text_feat    (text queries)
      K = W_k @ img_feat     (image keys)
      V = W_v @ img_feat     (image values)
      attention = softmax(Q @ K^T / sqrt(d))
      attended_img = attention @ V
      output = LayerNorm(text_feat + attended_img)
    """
    def __init__(self, text_dim: int, img_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = text_dim // num_heads
        assert text_dim % num_heads == 0, "text_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(text_dim, text_dim)
        self.k_proj = nn.Linear(img_dim, text_dim)
        self.v_proj = nn.Linear(img_dim, text_dim)
        self.out_proj = nn.Linear(text_dim, text_dim)
        self.norm = nn.LayerNorm(text_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_dim = text_dim

    def forward(self, text_feat: torch.Tensor, img_feat: torch.Tensor) -> torch.Tensor:
        # text_feat: (B, text_dim)  →  treat as 1-token sequence
        # img_feat:  (B, img_dim)   →  treat as 1-token sequence
        # For richer attention, we reshape to add a sequence dimension
        B = text_feat.size(0)

        # Add seq_len=1 dimension
        q = self.q_proj(text_feat).unsqueeze(1)  # (B, 1, text_dim)
        k = self.k_proj(img_feat).unsqueeze(1)   # (B, 1, text_dim)
        v = self.v_proj(img_feat).unsqueeze(1)   # (B, 1, text_dim)

        # Multi-head reshape
        q = q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention
        scale = math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (B, heads, 1, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, -1)  # (B, text_dim)
        out = self.out_proj(out)

        # Residual + norm
        return self.norm(text_feat + out)


class CrossModalAttentionDeep(nn.Module):
    """
    Enhanced cross-modal attention with multiple stacked layers.
    Each layer has its own Q/K/V projections and residual connections.
    """
    def __init__(self, text_dim: int, img_dim: int, num_heads: int = 4,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossModalAttention(text_dim, img_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.output_dim = text_dim

    def forward(self, text_feat: torch.Tensor, img_feat: torch.Tensor) -> torch.Tensor:
        out = text_feat
        for layer in self.layers:
            out = layer(out, img_feat)
        return out


class GatedFusion(nn.Module):
    """
    Gated Multimodal Fusion (adapted from Neverova et al., 2016).

    Learns a gating mechanism that dynamically weights the contribution
    of each modality based on input content.

    gate = sigmoid(W_g @ [text_feat; img_feat])
    output = gate * text_proj + (1 - gate) * img_proj
    """
    def __init__(self, text_dim: int, img_dim: int, hidden_dim: int = 256):
        super().__init__()
        combined = text_dim + img_dim
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.img_proj = nn.Linear(img_dim, hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(combined, hidden_dim),
            nn.Sigmoid(),
        )
        self.output_dim = hidden_dim

    def forward(self, text_feat: torch.Tensor, img_feat: torch.Tensor) -> torch.Tensor:
        combined = torch.cat((text_feat, img_feat), dim=1)
        g = self.gate(combined)
        t = self.text_proj(text_feat)
        i = self.img_proj(img_feat)
        return g * t + (1 - g) * i


class BilinearFusion(nn.Module):
    """
    Bilinear pooling fusion — captures second-order interactions
    between modalities (Tsai et al., 2017).
    """
    def __init__(self, text_dim: int, img_dim: int, output_dim: int = 256):
        super().__init__()
        self.bilinear = nn.Bilinear(text_dim, img_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(0.3)
        self.output_dim = output_dim

    def forward(self, text_feat: torch.Tensor, img_feat: torch.Tensor) -> torch.Tensor:
        out = self.bilinear(text_feat, img_feat)
        out = F.relu(out)
        out = self.dropout(out)
        return self.norm(out)


# ═══════════════════════════════════════════════════════════════════════
#  Full Classifiers
# ═══════════════════════════════════════════════════════════════════════

class MultimodalClassifier(nn.Module):
    """
    Unified multimodal classifier — supports multiple fusion strategies.

    fusion_type:
        'early'               — concat (baseline)
        'cross_attention'      — multi-head cross-modal attention
        'cross_attention_deep' — stacked cross-modal attention
        'gated'               — gated fusion (Neverova et al., 2016)
        'bilinear'            — bilinear pooling (Tsai et al., 2017)
    """

    FUSION_MAP = {
        "early": EarlyFusion,
        "cross_attention": CrossModalAttention,
        "cross_attention_deep": CrossModalAttentionDeep,
        "gated": GatedFusion,
        "bilinear": BilinearFusion,
    }

    def __init__(
        self,
        embedding_matrix: torch.Tensor,
        text_hidden_dim: int = 128,
        text_encoder: str = "bilstm",
        img_hidden_dim: int = 256,
        img_backbone: str = "vgg16",
        fusion_type: str = "early",
        freeze_vgg: bool = True,
    ):
        super().__init__()
        self.fusion_type = fusion_type

        # Text branch
        self.text_branch = TextBranch(
            embedding_matrix, hidden_dim=text_hidden_dim, encoder_type=text_encoder
        )

        # Image branch
        if img_backbone == "resnet50":
            self.image_branch = ImageBranchResNet(img_hidden_dim, freeze_backbone=freeze_vgg)
        else:
            self.image_branch = ImageBranch(img_hidden_dim, freeze_backbone=freeze_vgg)

        # Fusion
        fusion_cls = self.FUSION_MAP[fusion_type]
        if fusion_type in ("cross_attention", "cross_attention_deep"):
            self.fusion = fusion_cls(self.text_branch.output_dim, self.image_branch.output_dim)
        elif fusion_type == "bilinear":
            self.fusion = fusion_cls(self.text_branch.output_dim, self.image_branch.output_dim)
        elif fusion_type == "gated":
            self.fusion = fusion_cls(self.text_branch.output_dim, self.image_branch.output_dim)
        else:
            self.fusion = fusion_cls(self.text_branch.output_dim, self.image_branch.output_dim)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion.output_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, text: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        text_feat = self.text_branch(text)
        img_feat = self.image_branch(image)
        fused = self.fusion(text_feat, img_feat)
        return self.classifier(fused).squeeze(1)

    def get_fused_features(self, text: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """Extract fused feature vector (before classifier) — for visualization."""
        with torch.no_grad():
            text_feat = self.text_branch(text)
            img_feat = self.image_branch(image)
            return self.fusion(text_feat, img_feat)

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}
