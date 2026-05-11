"""Integration test: verify training runs for 1 epoch with mock data."""

import pytest
import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestTrainingIntegration:
    """End-to-end training tests with mock data."""

    def test_one_epoch_training(self, mock_data_dir, tmp_path):
        """Train for 1 epoch with mock data and verify checkpoint is saved."""
        from core.dataset import MultiOFFDataset, load_glove_embeddings
        from core.models import MultimodalClassifier
        from torch.utils.data import DataLoader
        import torch.nn as nn

        # Setup
        train_csv = os.path.join(mock_data_dir, "train.csv")
        val_csv = os.path.join(mock_data_dir, "val.csv")
        img_dir = os.path.join(mock_data_dir, "images")

        vocab, embeddings = load_glove_embeddings("/nonexistent/glove.txt", 50)

        train_ds = MultiOFFDataset(train_csv, img_dir, vocab)
        val_ds = MultiOFFDataset(val_csv, img_dir, vocab)

        train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

        model = MultimodalClassifier(
            embedding_matrix=embeddings,
            text_hidden_dim=32,
            img_hidden_dim=32,
            fusion_type="early",
        )

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        # Train 1 epoch
        model.train()
        for batch in train_loader:
            logits = model(batch["text"], batch["image"])
            loss = criterion(logits, batch["label"])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Validate
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                logits = model(batch["text"], batch["image"])
                assert logits.shape[0] > 0

        # Save checkpoint
        ckpt_path = str(tmp_path / "test_model.pth")
        torch.save({
            "model_state_dict": model.state_dict(),
            "args": {"fusion": "early", "text_hidden": 32, "img_hidden": 32},
            "glove_dim": 50,
            "vocab_size": len(vocab),
        }, ckpt_path)
        assert os.path.exists(ckpt_path)

    def test_loss_decreases(self, mock_data_dir):
        """Verify loss decreases over a few mini-batches."""
        from core.dataset import MultiOFFDataset, load_glove_embeddings
        from core.models import MultimodalClassifier
        from torch.utils.data import DataLoader
        import torch.nn as nn

        train_csv = os.path.join(mock_data_dir, "train.csv")
        img_dir = os.path.join(mock_data_dir, "images")

        vocab, embeddings = load_glove_embeddings("/nonexistent/glove.txt", 50)
        ds = MultiOFFDataset(train_csv, img_dir, vocab)
        loader = DataLoader(ds, batch_size=4, shuffle=True)

        model = MultimodalClassifier(
            embedding_matrix=embeddings,
            text_hidden_dim=32,
            img_hidden_dim=32,
            fusion_type="early",
        )
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        losses = []
        model.train()
        for i, batch in enumerate(loader):
            if i >= 5:
                break
            logits = model(batch["text"], batch["image"])
            loss = criterion(logits, batch["label"])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())

        # With 5 steps on the same data, loss should generally decrease
        # (not strictly monotonic, but first > last is very likely)
        assert losses[-1] < losses[0], f"Loss did not decrease: {losses}"
