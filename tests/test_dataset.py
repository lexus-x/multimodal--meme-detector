"""Tests for dataset loading and mock data generation."""

import os
import pytest
import torch
import pandas as pd

from core.dataset import (
    MultiOFFDataset,
    load_glove_embeddings,
    create_mock_dataset,
    VGG16_TRANSFORM,
    TRAIN_AUGMENT,
)


class TestMockDataset:
    """Test mock dataset generation."""

    def test_creates_files(self, tmp_path):
        output_dir = str(tmp_path / "mock")
        create_mock_dataset(output_dir, n_samples=20)
        assert os.path.exists(os.path.join(output_dir, "train.csv"))
        assert os.path.exists(os.path.join(output_dir, "val.csv"))
        assert os.path.exists(os.path.join(output_dir, "test.csv"))
        assert os.path.isdir(os.path.join(output_dir, "images"))

    def test_correct_sample_count(self, tmp_path):
        output_dir = str(tmp_path / "mock")
        create_mock_dataset(output_dir, n_samples=30)
        train_df = pd.read_csv(os.path.join(output_dir, "train.csv"))
        val_df = pd.read_csv(os.path.join(output_dir, "val.csv"))
        test_df = pd.read_csv(os.path.join(output_dir, "test.csv"))
        total = len(train_df) + len(val_df) + len(test_df)
        assert total == 30

    def test_csv_has_required_columns(self, tmp_path):
        output_dir = str(tmp_path / "mock")
        create_mock_dataset(output_dir, n_samples=10)
        df = pd.read_csv(os.path.join(output_dir, "train.csv"))
        assert "image_name" in df.columns
        assert "sentence" in df.columns
        assert "label" in df.columns

    def test_images_created(self, tmp_path):
        output_dir = str(tmp_path / "mock")
        create_mock_dataset(output_dir, n_samples=10)
        images_dir = os.path.join(output_dir, "images")
        images = os.listdir(images_dir)
        assert len(images) == 10
        assert all(f.endswith(".jpg") for f in images)

    def test_labels_are_binary(self, tmp_path):
        output_dir = str(tmp_path / "mock")
        create_mock_dataset(output_dir, n_samples=30)
        df = pd.read_csv(os.path.join(output_dir, "train.csv"))
        assert set(df["label"].unique()).issubset({0, 1})


class TestMultiOFFDataset:
    """Test the MultiOFFDataset PyTorch Dataset."""

    def test_dataset_length(self, mock_data_dir, vocab_and_embeddings):
        vocab, _ = vocab_and_embeddings
        csv_path = os.path.join(mock_data_dir, "train.csv")
        img_dir = os.path.join(mock_data_dir, "images")
        ds = MultiOFFDataset(csv_path, img_dir, vocab)
        df = pd.read_csv(csv_path)
        assert len(ds) == len(df)

    def test_getitem_returns_dict(self, mock_data_dir, vocab_and_embeddings):
        vocab, _ = vocab_and_embeddings
        csv_path = os.path.join(mock_data_dir, "train.csv")
        img_dir = os.path.join(mock_data_dir, "images")
        ds = MultiOFFDataset(csv_path, img_dir, vocab)
        item = ds[0]
        assert "image" in item
        assert "text" in item
        assert "label" in item
        assert "sentence" in item

    def test_image_shape(self, mock_data_dir, vocab_and_embeddings):
        vocab, _ = vocab_and_embeddings
        csv_path = os.path.join(mock_data_dir, "train.csv")
        img_dir = os.path.join(mock_data_dir, "images")
        ds = MultiOFFDataset(csv_path, img_dir, vocab)
        item = ds[0]
        assert item["image"].shape == (3, 224, 224)

    def test_text_tensor_shape(self, mock_data_dir, vocab_and_embeddings):
        vocab, _ = vocab_and_embeddings
        csv_path = os.path.join(mock_data_dir, "train.csv")
        img_dir = os.path.join(mock_data_dir, "images")
        ds = MultiOFFDataset(csv_path, img_dir, vocab, max_len=50)
        item = ds[0]
        assert item["text"].shape == (50,)
        assert item["text"].dtype == torch.long

    def test_label_is_float(self, mock_data_dir, vocab_and_embeddings):
        vocab, _ = vocab_and_embeddings
        csv_path = os.path.join(mock_data_dir, "train.csv")
        img_dir = os.path.join(mock_data_dir, "images")
        ds = MultiOFFDataset(csv_path, img_dir, vocab)
        item = ds[0]
        assert item["label"].dtype == torch.float32

    def test_with_augmentation(self, mock_data_dir, vocab_and_embeddings):
        vocab, _ = vocab_and_embeddings
        csv_path = os.path.join(mock_data_dir, "train.csv")
        img_dir = os.path.join(mock_data_dir, "images")
        ds = MultiOFFDataset(csv_path, img_dir, vocab, transform=TRAIN_AUGMENT)
        item = ds[0]
        assert item["image"].shape == (3, 224, 224)

    def test_missing_image_fallback(self, tmp_path, vocab_and_embeddings):
        """Test that missing images produce a gray placeholder."""
        vocab, _ = vocab_and_embeddings
        # Create CSV with non-existent images
        df = pd.DataFrame({
            "image_name": ["nonexistent.jpg"],
            "sentence": ["test sentence"],
            "label": [0],
        })
        csv_path = str(tmp_path / "test.csv")
        df.to_csv(csv_path, index=False)
        ds = MultiOFFDataset(csv_path, str(tmp_path), vocab)
        item = ds[0]
        assert item["image"].shape == (3, 224, 224)


class TestGloveEmbeddings:
    """Test GloVe embedding loading."""

    def test_load_with_missing_file(self):
        """Should return random embeddings when file doesn't exist."""
        vocab, matrix = load_glove_embeddings("/nonexistent/path.txt", 50)
        assert "<pad>" in vocab
        assert "<unk>" in vocab
        assert matrix.shape[1] == 50
        # Should have at least pad + unk + random words
        assert matrix.shape[0] >= 502

    def test_pad_token_is_zero(self):
        vocab, matrix = load_glove_embeddings("/nonexistent/path.txt", 50)
        assert torch.all(matrix[0] == 0)

    def test_vocab_indices(self):
        vocab, _ = load_glove_embeddings("/nonexistent/path.txt", 50)
        assert vocab["<pad>"] == 0
        assert vocab["<unk>"] == 1
