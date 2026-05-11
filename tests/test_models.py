"""Tests for model architectures and fusion strategies."""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.models import (
    TextBranch,
    ImageBranch,
    ImageBranchResNet,
    EarlyFusion,
    CrossModalAttention,
    CrossModalAttentionDeep,
    GatedFusion,
    BilinearFusion,
    MultimodalClassifier,
)


class TestTextBranch:
    """Test TextBranch encoder variants."""

    def test_bilstm_forward(self, embedding_matrix):
        model = TextBranch(embedding_matrix, hidden_dim=64, encoder_type="bilstm")
        x = torch.randint(0, 100, (4, 50))  # batch=4, seq_len=50
        out = model(x)
        assert out.shape == (4, 128)  # 64 * 2 (bidirectional)

    def test_lstm_forward(self, embedding_matrix):
        model = TextBranch(embedding_matrix, hidden_dim=64, encoder_type="lstm")
        x = torch.randint(0, 100, (4, 50))
        out = model(x)
        assert out.shape == (4, 64)

    def test_cnn_forward(self, embedding_matrix):
        model = TextBranch(embedding_matrix, hidden_dim=64, encoder_type="cnn")
        x = torch.randint(0, 100, (4, 50))
        out = model(x)
        assert out.shape == (4, 192)  # 64 * 3 (three kernel sizes)

    def test_invalid_encoder(self, embedding_matrix):
        with pytest.raises(ValueError, match="encoder_type must be one of"):
            TextBranch(embedding_matrix, encoder_type="transformer")


class TestImageBranch:
    """Test image feature extractors."""

    def test_vgg16_forward(self):
        model = ImageBranch(hidden_dim=128, freeze_backbone=True)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.shape == (2, 128)

    def test_resnet50_forward(self):
        model = ImageBranchResNet(hidden_dim=128, freeze_backbone=True)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.shape == (2, 128)

    def test_output_dim_matches(self):
        model = ImageBranch(hidden_dim=256)
        assert model.output_dim == 256


class TestFusionStrategies:
    """Test each fusion strategy's forward pass."""

    @pytest.fixture
    def features(self):
        """Mock text and image features."""
        text_feat = torch.randn(4, 256)
        img_feat = torch.randn(4, 256)
        return text_feat, img_feat

    def test_early_fusion(self, features):
        text_feat, img_feat = features
        fusion = EarlyFusion(256, 256)
        out = fusion(text_feat, img_feat)
        assert out.shape == (4, 512)
        assert fusion.output_dim == 512

    def test_cross_modal_attention(self, features):
        text_feat, img_feat = features
        fusion = CrossModalAttention(256, 256, num_heads=4)
        out = fusion(text_feat, img_feat)
        assert out.shape == (4, 256)
        assert fusion.output_dim == 256

    def test_cross_modal_attention_deep(self, features):
        text_feat, img_feat = features
        fusion = CrossModalAttentionDeep(256, 256, num_heads=4, num_layers=2)
        out = fusion(text_feat, img_feat)
        assert out.shape == (4, 256)

    def test_gated_fusion(self, features):
        text_feat, img_feat = features
        fusion = GatedFusion(256, 256, hidden_dim=128)
        out = fusion(text_feat, img_feat)
        assert out.shape == (4, 128)
        assert fusion.output_dim == 128

    def test_bilinear_fusion(self, features):
        text_feat, img_feat = features
        fusion = BilinearFusion(256, 256, output_dim=128)
        out = fusion(text_feat, img_feat)
        assert out.shape == (4, 128)


class TestMultimodalClassifier:
    """Test the full classifier with different fusion strategies."""

    @pytest.fixture
    def inputs(self):
        text = torch.randint(0, 100, (2, 50))
        image = torch.randn(2, 3, 224, 224)
        return text, image

    @pytest.mark.parametrize("fusion_type", [
        "early", "cross_attention", "gated", "bilinear"
    ])
    def test_classifier_forward(self, embedding_matrix, inputs, fusion_type):
        text, image = inputs
        model = MultimodalClassifier(
            embedding_matrix=embedding_matrix,
            text_hidden_dim=64,
            img_hidden_dim=64,
            fusion_type=fusion_type,
        )
        model.eval()
        with torch.no_grad():
            logits = model(text, image)
        assert logits.shape == (2,)

    def test_sigmoid_output_range(self, embedding_matrix, inputs):
        text, image = inputs
        model = MultimodalClassifier(embedding_matrix=embedding_matrix)
        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(text, image))
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_count_parameters(self, embedding_matrix):
        model = MultimodalClassifier(embedding_matrix=embedding_matrix)
        params = model.count_parameters()
        assert "total" in params
        assert "trainable" in params
        assert "frozen" in params
        assert params["total"] > 0
        assert params["total"] == params["trainable"] + params["frozen"]

    def test_get_fused_features(self, embedding_matrix, inputs):
        text, image = inputs
        model = MultimodalClassifier(embedding_matrix=embedding_matrix)
        model.eval()
        features = model.get_fused_features(text, image)
        assert features.dim() == 2
        assert features.shape[0] == 2

    def test_backward_pass(self, embedding_matrix, inputs):
        """Verify gradients flow through the model."""
        text, image = inputs
        model = MultimodalClassifier(embedding_matrix=embedding_matrix)
        model.train()
        labels = torch.tensor([0.0, 1.0])
        logits = model(text, image)
        loss = torch.nn.BCEWithLogitsLoss()(logits, labels)
        loss.backward()
        # Check at least some parameters have gradients
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in model.parameters() if p.requires_grad)
        assert has_grad, "No gradients found"
