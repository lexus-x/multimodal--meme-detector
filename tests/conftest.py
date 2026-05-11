"""Shared test fixtures for the multimodal meme detector."""

import os
import sys
import pytest
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def embedding_matrix():
    """Small random embedding matrix for testing (vocab_size=100, dim=50)."""
    return torch.randn(100, 50)


@pytest.fixture
def mock_data_dir(tmp_path):
    """Create a temporary mock dataset."""
    from core.dataset import create_mock_dataset
    output_dir = str(tmp_path / "mock_data")
    create_mock_dataset(output_dir, n_samples=20)
    return output_dir


@pytest.fixture
def vocab_and_embeddings():
    """Return a small vocab dict and embedding matrix."""
    vocab = {"<pad>": 0, "<unk>": 1}
    embeddings = [torch.zeros(50), torch.randn(50)]
    for i in range(50):
        vocab[f"word_{i}"] = i + 2
        embeddings.append(torch.randn(50))
    return vocab, torch.stack(embeddings)
