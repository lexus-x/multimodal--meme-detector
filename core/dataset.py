"""
Multimodal Meme Offensive Content Detection — Dataset

Handles MultiOFF dataset loading, GloVe embeddings, and
image/text preprocessing for PyTorch DataLoader consumption.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


# ── Image Transforms ─────────────────────────────────────────────────

VGG16_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

TRAIN_AUGMENT = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

RESNET_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ── GloVe Loader ─────────────────────────────────────────────────────

def load_glove_embeddings(glove_path: str, embedding_dim: int = 50
                          ) -> tuple[dict[str, int], torch.Tensor]:
    """Load GloVe embeddings. Returns (vocab, embedding_matrix)."""
    vocab: dict[str, int] = {"<pad>": 0, "<unk>": 1}
    embeddings: list[np.ndarray] = [
        np.zeros(embedding_dim),                             # <pad>
        np.random.normal(scale=0.6, size=(embedding_dim,)),  # <unk>
    ]
    idx = 2

    if os.path.exists(glove_path):
        print(f"Loading GloVe ({embedding_dim}d) from {glove_path} ...")
        with open(glove_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                word = parts[0]
                vec = np.asarray(parts[1:], dtype="float32")
                if vec.shape[0] != embedding_dim:
                    continue
                vocab[word] = idx
                embeddings.append(vec)
                idx += 1
        print(f"  Loaded {idx:,} words")
    else:
        print(f"  ⚠ {glove_path} not found — using random embeddings (demo mode)")
        for i in range(500):
            vocab[f"word_{i}"] = idx
            embeddings.append(np.random.normal(scale=0.6, size=(embedding_dim,)))
            idx += 1

    matrix = np.vstack(embeddings).astype("float32")
    return vocab, torch.tensor(matrix)


# ── MultiOFF Dataset ─────────────────────────────────────────────────

class MultiOFFDataset(Dataset):
    """PyTorch Dataset for MultiOFF offensive meme classification.

    Expects CSV with columns: image_name, sentence, label
    """

    PAD_IDX = 0
    UNK_IDX = 1

    def __init__(self, csv_file: str, img_dir: str, vocab: dict,
                 max_len: int = 50, transform=None):
        self.df = pd.read_csv(csv_file).reset_index(drop=True)
        self.img_dir = img_dir
        self.vocab = vocab
        self.max_len = max_len
        self.transform = transform or VGG16_TRANSFORM

    def __len__(self) -> int:
        return len(self.df)

    def _tokenize(self, sentence: str) -> torch.Tensor:
        tokens = str(sentence).lower().split()
        indices = [self.vocab.get(t, self.UNK_IDX) for t in tokens]
        if len(indices) < self.max_len:
            indices += [self.PAD_IDX] * (self.max_len - len(indices))
        else:
            indices = indices[:self.max_len]
        return torch.tensor(indices, dtype=torch.long)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        # Image
        img_path = os.path.join(self.img_dir, str(row["image_name"]))
        try:
            image = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, OSError):
            image = Image.new("RGB", (224, 224), color=(128, 128, 128))
        image = self.transform(image)

        # Text
        text = self._tokenize(row["sentence"])

        # Label
        label = torch.tensor(row["label"], dtype=torch.float32)

        return {"image": image, "text": text, "label": label, "sentence": str(row["sentence"])}


# ── Mock Dataset ─────────────────────────────────────────────────────

def create_mock_dataset(output_dir: str = "mock_data", n_samples: int = 60):
    """Generate a tiny mock dataset for pipeline smoke-testing."""
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

    sentences = [
        "This is a perfectly normal and friendly meme",
        "What a beautiful day to be alive",
        "I love my cat so much",
        "Programming is fun when it works",
        "The weather is nice today",
        "Having coffee with friends",
        "This movie was absolutely terrible",
        "I hate when people do that",
        "You are the worst person ever",
        "Go away nobody likes you",
        "This is so offensive and rude",
        "What an idiot this guy is",
        "Shut up you fool",
        "You disgust me completely",
        "Such a stupid and dumb idea",
    ]

    for i in range(n_samples):
        img = Image.new("RGB", (224, 224),
                        color=(i * 5 % 256, i * 3 % 256, i * 7 % 256))
        img.save(os.path.join(output_dir, "images", f"img_{i}.jpg"))

    # Correlate label with sentence sentiment so the model can actually learn it (hit >90% accuracy)
    labels = []
    for s in [sentences[i % len(sentences)] for i in range(n_samples)]:
        # Assign 1 (Offensive) if it contains negative words
        if any(w in s for w in ["terrible", "hate", "worst", "away", "offensive", "idiot", "fool", "disgust", "stupid"]):
            labels.append(1)
        else:
            labels.append(0)

    data = {
        "image_name": [f"img_{i}.jpg" for i in range(n_samples)],
        "sentence": [sentences[i % len(sentences)] for i in range(n_samples)],
        "label": labels,
    }
    df = pd.DataFrame(data)
    n_train = int(n_samples * 0.7)
    n_val = int(n_samples * 0.15)

    df.iloc[:n_train].to_csv(os.path.join(output_dir, "train.csv"), index=False)
    df.iloc[n_train:n_train + n_val].to_csv(os.path.join(output_dir, "val.csv"), index=False)
    df.iloc[n_train + n_val:].to_csv(os.path.join(output_dir, "test.csv"), index=False)

    print(f"  Mock dataset: {output_dir}/ ({n_samples} samples)")
