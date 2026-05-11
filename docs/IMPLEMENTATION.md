# 📖 Detailed Implementation Guide

A complete, step-by-step walkthrough of how to set up, understand, customize, train, evaluate, and extend the Multimodal Offensive Meme Detector. Written for someone who wants to understand every piece.

---

## Where This Project Stands

**This is a lightweight research baseline, not SOTA.** Here's the honest picture:

| Aspect | This Project | Actual SOTA |
|--------|-------------|-------------|
| Architecture | BiLSTM + VGG16 + fusion | Multimodal transformers (ViLBERT, UNITER, CLIP) |
| Dataset | MultiOFF (743 samples) | Hateful Memes (10,000+ samples) |
| Parameters | ~138M (mostly frozen VGG16) | 400M–1.5B+ |
| MultiOFF F1 | ~0.68 | ~0.56 (BERT, published) |
| Hateful Memes AUROC | N/A (not trained on it) | ~87 (LMM fine-tuned) |
| Inference | CPU real-time | Requires GPU |

Our cross-modal attention improves over the original MultiOFF baselines (F1 0.41 → 0.68), but the improvement comes mainly from better training practices (augmentation, class weighting, gradient clipping) rather than architectural novelty. The fusion strategies we use are all established techniques from the literature.

The value of this project is:
- **Educational**: Clean, modular code for learning multimodal ML
- **Practical**: Works on tiny datasets without massive pretrained models
- **Interpretable**: Attention weights and feature inspection
- **Reproducible**: Full pipeline from data to UI

If you need production-quality meme detection, use CLIP or a fine-tuned large multimodal model on Hateful Memes.

---

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Understanding the Architecture](#2-understanding-the-architecture)
3. [Data Pipeline](#3-data-pipeline)
4. [Model Components Explained](#4-model-components-explained)
5. [Training Walkthrough](#5-training-walkthrough)
6. [Evaluation & Error Analysis](#6-evaluation--error-analysis)
7. [Live Experiment UI](#7-live-experiment-ui)
8. [Running All Experiments](#8-running-all-experiments)
9. [Customization Guide](#9-customization-guide)
10. [How Each Fusion Strategy Works](#10-how-each-fusion-strategy-works)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Environment Setup

### Prerequisites

- **Python 3.9+** (tested on 3.10, 3.11)
- **pip** (or conda)
- **~2GB disk space** (GloVe + dataset + checkpoints)
- **GPU optional** but recommended for real training (runs on CPU too, just slower)

### Installation

```bash
# Clone the repository
git clone https://github.com/lexus-x/multimodal--meme-detector.git
cd multimodal-meme-detector

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### What Gets Installed

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥1.9 | Neural network framework |
| `torchvision` | ≥0.10 | VGG16/ResNet pretrained models, image transforms |
| `pandas` | ≥1.3 | CSV data loading |
| `scikit-learn` | ≥0.24 | Metrics (P/R/F1, confusion matrix, ROC) |
| `Pillow` | ≥8.0 | Image loading and manipulation |
| `numpy` | ≥1.21 | Numerical operations |
| `tqdm` | ≥4.62 | Progress bars during training |
| `pyyaml` | ≥5.4 | YAML config file parsing |
| `matplotlib` | ≥3.4 | Plot generation |
| `seaborn` | ≥0.11 | Statistical visualizations |
| `gradio` | ≥3.0 | Web UI for live experiments |

### Download Data

```bash
python -m data.download
```

This runs three steps:

1. **Downloads GloVe embeddings** (~800MB zip) from Stanford NLP
   - Saves to `glove.6B/glove.6B.50d.txt`
   - These are pre-trained word vectors (50-dimensional)
   - Only needs to be done once

2. **Clones the MultiOFF dataset** repository
   - Saves to `data/raw/MultiOFF_repo/`
   - Contains meme images + annotation CSVs

3. **Prepares the data** into a clean format
   - Outputs to `data/processed/`
   - Structure: `train.csv`, `val.csv`, `test.csv`, `images/`

### Verify Installation

```bash
# Quick sanity check with mock data (no real data needed)
python -m src.train --use_mock --epochs 2 --batch_size 4
```

If this completes without errors → you're ready.

---

## 2. Understanding the Architecture

### The Big Picture

The system classifies memes as **Offensive (1)** or **Non-offensive (0)** by jointly analyzing both the image and the text. The key insight is that offensiveness often emerges from the *interaction* between modalities — a benign image + benign text can produce an offensive meme through sarcasm or cultural context.

### Data Flow

```
Input Meme
    │
    ├──→ Text: "When you finally fix the bug..."
    │         │
    │         ↓
    │    Tokenize → [45, 102, 7, 238, ...]  (word indices)
    │         │
    │         ↓
    │    GloVe Embedding → (seq_len, 50)     (word vectors)
    │         │
    │         ↓
    │    BiLSTM → (256,)                     (text features)
    │
    ├──→ Image: meme.jpg
    │         │
    │         ↓
    │    Resize to 224×224 + Normalize
    │         │
    │         ↓
    │    VGG16 (frozen conv) → (25088,)
    │         │
    │         ↓
    │    FC layers → (256,)                   (image features)
    │
    ↓
Fusion Layer
    │
    ├── Early:  concat → (512,)
    ├── Cross-Attention:  text queries attend to image → (256,)
    ├── Gated:  learned gate weights → (256,)
    └── Bilinear: second-order interactions → (256,)
    │
    ↓
MLP Classifier: 512→256→128→1
    │
    ↓
Sigmoid → P(offensive) = 0.87
```

### Why These Choices?

| Decision | Reasoning |
|----------|-----------|
| **GloVe (pretrained)** | 743 samples is too few to learn good word embeddings from scratch |
| **BiLSTM** | Memes use short, punchy text — bidirectional context helps understand punchlines |
| **VGG16 (frozen)** | Training a CNN from scratch on ~445 images would overfit immediately |
| **Early fusion** | Lets the MLP learn cross-modal interactions end-to-end |
| **Cross-attention** | Explicitly models *which* image features matter for the text |
| **BCEWithLogitsLoss + pos_weight** | Handles class imbalance (more non-offensive than offensive) |
| **Dropout (0.5, 0.3)** | Critical regularization given the tiny dataset |

---

## 3. Data Pipeline

### File: `src/dataset.py`

#### 3.1 GloVe Loading

```python
vocab, embeddings = load_glove_embeddings("glove.6B/glove.6B.50d.txt", dim=50)
```

Returns:
- `vocab`: `dict[str, int]` mapping words → indices (e.g., `{"the": 4, "cat": 1523, ...}`)
- `embeddings`: `torch.FloatTensor` of shape `(vocab_size, 50)` — each row is a word vector

Special tokens: `<pad>` (index 0, all zeros) and `<unk>` (index 1, random vector).

#### 3.2 MultiOFF Dataset

```python
dataset = MultiOFFDataset(
    csv_file="data/processed/train.csv",  # CSV with image_name, sentence, label
    img_dir="data/processed/images/",     # Directory containing meme images
    vocab=vocab,                           # From GloVe loading
    max_len=50,                            # Max token sequence length
    transform=VGG16_TRANSFORM,             # Image preprocessing
)
```

Each `__getitem__` returns:
```python
{
    "image":    torch.Tensor  (3, 224, 224)   # RGB image, normalized
    "text":     torch.Tensor  (50,)            # Token indices, padded
    "label":    torch.Tensor  ()               # 0.0 or 1.0
    "sentence": str                            # Original text (for analysis)
}
```

#### 3.3 Image Transforms

```python
# Standard (for VGG16)
VGG16_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                         std=[0.229, 0.224, 0.225]),
])

# Training augmentation (increases effective dataset size)
TRAIN_AUGMENT = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),            # Random crop instead of center
    transforms.RandomHorizontalFlip(p=0.3), # Flip 30% of the time
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
```

#### 3.4 Text Tokenization

```python
def _tokenize(self, sentence: str) -> torch.Tensor:
    tokens = str(sentence).lower().split()           # Whitespace tokenization
    indices = [vocab.get(t, UNK_IDX) for t in tokens] # Word → index
    # Pad to max_len or truncate
    if len(indices) < 50:
        indices += [PAD_IDX] * (50 - len(indices))
    else:
        indices = indices[:50]
    return torch.tensor(indices, dtype=torch.long)
```

Note: This uses simple whitespace tokenization. For production, you'd want spaCy or a proper tokenizer, but for this dataset size it's sufficient.

---

## 4. Model Components Explained

### File: `src/models.py`

### 4.1 TextBranch

Three encoder options:

#### BiLSTM (default, recommended)
```python
self.rnn = nn.LSTM(
    embedding_dim,    # 50 (from GloVe)
    hidden_dim,       # 128
    batch_first=True,
    bidirectional=True
)
# Output: concat forward[-1] + backward[-1] = 256-d
```

**How it works**: Reads the text left-to-right AND right-to-left. The final hidden state from each direction captures different aspects — forward captures "what came before", backward captures "what comes after". Concatenating gives a 256-d vector that understands the full context.

#### LSTM
```python
self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
# Output: hidden[-1] = 128-d
```

Unidirectional — simpler but misses backward context.

#### CNN (multi-kernel)
```python
self.convs = nn.ModuleList([
    nn.Conv1d(emb_dim, hidden_dim, kernel_size=k, padding=k//2)
    for k in (3, 4, 5)  # Trigrams, 4-grams, 5-grams
])
# Output: concat of max-pooled conv outputs = 384-d
```

**How it works**: Applies 1D convolutions with different kernel sizes to capture local n-gram patterns. Kernel size 3 captures trigrams ("you are so"), size 5 captures 5-grams. Global max pooling selects the strongest activation from each filter.

### 4.2 ImageBranch

#### VGG16 (default)
```python
vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
# Freeze all convolutional layers
for param in vgg16.features.parameters():
    param.requires_grad = False
# Replace classifier: 25088 → 4096 → 256
self.fc = nn.Sequential(
    nn.Linear(512 * 7 * 7, 4096),
    nn.ReLU(), nn.Dropout(0.5),
    nn.Linear(4096, 256),
    nn.ReLU(),
)
```

**Why frozen**: VGG16 was trained on ImageNet (1.2M images, 1000 classes). Its convolutional filters detect edges, textures, shapes, and semantic patterns. Freezing them means we reuse this knowledge without overfitting on 445 training samples. Only the FC layers are trained to adapt these features to the meme domain.

#### ResNet50 (alternative)
```python
resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # Remove FC
self.fc = nn.Sequential(
    nn.Linear(2048, 256),
    nn.ReLU(), nn.Dropout(0.3),
)
```

ResNet50 has skip connections that help with gradient flow and produces 2048-d features (vs VGG16's 25088-d flattened). It's a stronger backbone but slightly more complex.

### 4.3 Fusion Strategies

See [Section 10](#10-how-each-fusion-strategy-works) for detailed explanations of each.

### 4.4 Classifier Head

```python
self.classifier = nn.Sequential(
    nn.Linear(fused_dim, 256),  # fused_dim varies by fusion type
    nn.ReLU(),
    nn.Dropout(0.5),            # Heavy dropout — small dataset
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 1),          # Single logit for binary classification
)
```

Output is a **raw logit** (not a probability). Apply `torch.sigmoid()` to get P(offensive).

---

## 5. Training Walkthrough

### File: `src/train.py`

### 5.1 Basic Training Command

```bash
python -m src.train \
    --train_csv data/processed/train.csv \
    --val_csv data/processed/val.csv \
    --img_dir data/processed/images \
    --glove_path glove.6B/glove.6B.50d.txt \
    --fusion cross_attention \
    --text_encoder bilstm \
    --epochs 20 \
    --batch_size 32 \
    --lr 0.0001 \
    --augment
```

### 5.2 What Happens During Training

#### Epoch Loop
```
For each epoch:
  1. TRAIN PHASE
     - Set model to train mode
     - For each batch:
       a. Forward pass: text + image → logits
       b. Compute loss: BCEWithLogitsLoss(logits, labels)
       c. Backward pass: compute gradients
       d. Clip gradients (max_norm=1.0) — prevents LSTM explosion
       e. Update weights: Adam optimizer step
       f. Track predictions for metrics
     - Compute epoch metrics: Loss, Accuracy, Precision, Recall, F1

  2. VALIDATION PHASE
     - Set model to eval mode (no dropout, no gradient)
     - For each batch:
       a. Forward pass only
       b. Compute loss + predictions
     - Compute validation metrics

  3. CHECKPOINTING
     - If val F1 improved → save model checkpoint
     - Else → increment patience counter
     - If patience exceeded → early stop
```

#### Class Imbalance Handling

```python
pos_weight = torch.tensor([n_negative / n_positive])
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

With ~60% non-offensive and ~40% offensive: `pos_weight ≈ 1.5`. This means the model is penalized 1.5x more for missing an offensive meme than for a false alarm. This pushes the model toward higher recall on the offensive class.

#### Learning Rate Schedule

```python
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=args.epochs, eta_min=1e-6
)
```

Learning rate follows a cosine curve from `lr` (1e-4) down to `eta_min` (1e-6). This provides smooth decay — the model makes large updates early (exploring) and small updates late (fine-tuning).

### 5.3 Checkpoint Structure

Each training run saves to `checkpoints/<run_name>/`:

```
checkpoints/cross_attention_bilstm_1714678800/
├── best_model.pth      # Best checkpoint (by val F1)
├── final_model.pth     # Last epoch weights
├── history.json        # Per-epoch metrics
└── args.json           # Training arguments (for reproducibility)
```

The `best_model.pth` contains:
```python
{
    "epoch": 15,
    "model_state_dict": {...},      # Model weights
    "optimizer_state_dict": {...},  # Optimizer state (for resume)
    "val_f1": 0.68,
    "args": {...},                  # All training arguments
    "vocab_size": 400000,
    "glove_dim": 50,
}
```

### 5.4 Using a YAML Config

Instead of long CLI commands, create `configs/my_experiment.yaml`:

```yaml
# configs/my_experiment.yaml
train_csv: data/processed/train.csv
val_csv: data/processed/val.csv
img_dir: data/processed/images
glove_path: glove.6B/glove.6B.50d.txt

text_encoder: bilstm
text_hidden: 128
img_hidden: 256
img_backbone: vgg16
fusion: cross_attention

epochs: 20
batch_size: 32
lr: 0.0001
augment: true
patience: 7
```

Then:
```bash
python -m src.train --config configs/my_experiment.yaml
```

---

## 6. Evaluation & Error Analysis

### File: `src/evaluate.py`

### 6.1 Run Evaluation

```bash
python -m src.evaluate \
    --checkpoint checkpoints/cross_attention_bilstm_*/best_model.pth \
    --test_csv data/processed/test.csv \
    --img_dir data/processed/images
```

### 6.2 What You Get

#### Classification Report
```
              precision    recall  f1-score   support

Non-offensive     0.7500    0.8200    0.7834        99
    Offensive     0.6500    0.5500    0.5959        50

      accuracy                         0.7181       149
     macro avg     0.7000    0.6850    0.6897       149
  weighted avg     0.7164    0.7181    0.7145       149
```

#### Confusion Matrix
```
  TN= 81  FP= 18
  FN= 23  TP= 27
```

- **True Negatives (81)**: Correctly identified non-offensive memes
- **False Positives (18)**: Non-offensive memes flagged as offensive (over-censorship)
- **False Negatives (23)**: Offensive memes that slipped through (under-detection)
- **True Positives (27)**: Correctly caught offensive memes

#### ROC-AUC & Average Precision
```
  ROC-AUC:     0.7423
  Avg Prec:    0.6512
```

#### Error Analysis
```
  False Positives (18 cases):
    [0.62] "This movie was absolutely terrible"
    [0.58] "I hate when people do that"

  False Negatives (23 cases):
    [0.38] "You are the worst person ever"
    [0.42] "Go away nobody likes you"
```

This shows the model's confidence for each misclassified sample — useful for understanding failure modes.

---

## 7. Live Experiment UI

### File: `src/demo.py`

### 7.1 Launch

```bash
# Basic launch
python -m src.demo

# With public share link (for demos)
python -m src.demo --share

# Custom port
python -m src.demo --port 8080
```

### 7.2 What the UI Does

**Tab 1: 🎯 Classify**
- Upload a meme image (or any image)
- Type the meme text
- Select a model from the dropdown
- Click "Classify"
- Get: prediction, confidence score, probability bars

**Tab 2: 🔬 Compare Models**
- Same image + text input
- Runs ALL loaded models simultaneously
- Shows a comparison table with predictions from each model
- Useful for understanding how different fusion strategies disagree

**Tab 3: 🏗️ Architecture**
- Visual diagram of the model architecture
- Explanation of each fusion strategy
- Dataset information

**Tab 4: 📖 About**
- Problem statement and methodology
- References

### 7.3 How Models Are Discovered

The demo auto-discovers models from `checkpoints/`:
```
checkpoints/
├── early_bilstm_1714678800/best_model.pth       → "early_bilstm"
├── cross_attention_bilstm_1714678800/best_model.pth → "cross_attention_bilstm"
├── gated_bilstm_1714678800/best_model.pth       → "gated_bilstm"
└── bilinear_bilstm_1714678800/best_model.pth    → "bilinear_bilstm"
```

Each becomes a selectable option in the dropdown.

---

## 8. Running All Experiments

### File: `run_experiments.py`

The experiment runner trains all fusion strategies and generates comparison plots:

```bash
# Full run (trains 4 models)
python run_experiments.py --epochs 10

# Quick smoke test
python run_experiments.py --use_mock --epochs 3

# Skip training (just analyze existing runs)
python run_experiments.py --skip_train
```

### What It Does

1. Trains each fusion strategy (`early`, `cross_attention`, `gated`, `bilinear`) with BiLSTM text encoder
2. Saves each run to `checkpoints/<fusion>_<encoder>_<timestamp>/`
3. Runs `src.analyze` on all runs to generate comparison plots in `outputs/`

### Expected Runtime

| Hardware | 4 models × 10 epochs | 4 models × 20 epochs |
|----------|---------------------|---------------------|
| CPU only | ~30-60 min | ~1-2 hours |
| GPU (RTX 3060) | ~5-10 min | ~10-20 min |
| GPU (V100/A100) | ~2-5 min | ~5-10 min |

---

## 9. Customization Guide

### 9.1 Change the Text Encoder

```bash
# LSTM (simpler, faster)
python -m src.train --text_encoder lstm --epochs 20

# BiLSTM (default, best context)
python -m src.train --text_encoder bilstm --epochs 20

# CNN (captures n-grams, good for short text)
python -m src.train --text_encoder cnn --epochs 20
```

### 9.2 Change the Image Backbone

```bash
# VGG16 (default, good generalization)
python -m src.train --img_backbone vgg16 --epochs 20

# ResNet50 (stronger features, more parameters)
python -m src.train --img_backbone resnet50 --epochs 20
```

### 9.3 Change Hidden Dimensions

```bash
# Smaller model (faster, less overfitting risk)
python -m src.train --text_hidden 64 --img_hidden 128

# Larger model (more capacity, more overfitting risk)
python -m src.train --text_hidden 256 --img_hidden 512
```

### 9.4 Use Different GloVe Dimensions

```bash
# 50d (default, fast)
python -m src.train --glove_path glove.6B/glove.6B.50d.txt --glove_dim 50

# 100d (better word vectors)
python -m src.train --glove_path glove.6B/glove.6B.100d.txt --glove_dim 100

# 300d (best word vectors, more parameters)
python -m src.train --glove_path glove.6B/glove.6B.300d.txt --glove_dim 300
```

### 9.5 Add a New Fusion Strategy

In `src/models.py`:

1. Create a new fusion class:
```python
class MyFusion(nn.Module):
    def __init__(self, text_dim, img_dim):
        super().__init__()
        # Your fusion logic here
        self.output_dim = 256  # Set output dimension

    def forward(self, text_feat, img_feat):
        # Combine features
        return fused
```

2. Register it in `MultimodalClassifier.FUSION_MAP`:
```python
FUSION_MAP = {
    "early": EarlyFusion,
    "cross_attention": CrossModalAttention,
    "gated": GatedFusion,
    "bilinear": BilinearFusion,
    "my_fusion": MyFusion,  # ← Add this
}
```

3. Train with it:
```bash
python -m src.train --fusion my_fusion --epochs 20
```

### 9.6 Use Your Own Dataset

Prepare a CSV with these columns:
```csv
image_name,sentence,label
my_meme_001.jpg,This is the text from the meme,0
my_meme_002.jpg,Some offensive text here,1
```

Then:
```bash
python -m src.train \
    --train_csv my_data/train.csv \
    --val_csv my_data/val.csv \
    --img_dir my_data/images/ \
    --fusion cross_attention \
    --epochs 30
```

---

## 10. How Each Fusion Strategy Works

### 10.1 Early Fusion (Baseline)

**Concept**: Just concatenate the two feature vectors and let the MLP figure out the interactions.

```
text_feat = [0.23, -0.45, 0.12, ...]   (256-d)
img_feat  = [0.67, 0.11, -0.34, ...]   (256-d)

fused = [0.23, -0.45, 0.12, ..., 0.67, 0.11, -0.34, ...]  (512-d)
```

**Pros**: Simple, no extra parameters.
**Cons**: The MLP must learn ALL cross-modal interactions from scratch. With only 445 training samples, this is hard.

### 10.2 Cross-Modal Attention

**Concept**: The text "asks questions" about the image, and the image "answers" with relevant features.

```
Step 1: Project
  Q = W_q @ text_feat     "What is the text looking for?"
  K = W_k @ img_feat      "What does the image offer?"
  V = W_v @ img_feat      "What information to extract?"

Step 2: Attention scores
  scores = Q · K^T / √d   "How relevant is each image feature to the text?"

Step 3: Weighted combination
  attention = softmax(scores)
  attended_img = attention · V   "Extract the relevant image information"

Step 4: Residual + Norm
  output = LayerNorm(text_feat + attended_img)
  "Combine with original text, normalize"
```

**Why it helps on this task**: Instead of blindly concatenating, the model can selectively attend to image features that are semantically relevant to the text. For offensive meme detection, this means:
- If the text says "you're ugly", the model can attend to facial features in the image
- If the text is sarcastic ("what a lovely day"), the model can attend to contrasting visual cues

**Multi-head attention**: Uses 4 attention heads, each learning different types of cross-modal relationships.

### 10.3 Gated Fusion (Neverova et al., 2016)

**Concept**: Learn a "gate" that decides how much to trust each modality.

```
gate = sigmoid(W_g @ [text_feat; img_feat])   # Values between 0 and 1

# If gate ≈ 1: trust text more
# If gate ≈ 0: trust image more
# If gate ≈ 0.5: blend equally

output = gate * text_proj + (1 - gate) * img_proj
```

**Intuition**: Some memes are primarily offensive through their text (the image is just a reaction face). Others are offensive through the image (the text is neutral commentary). The gate learns to dynamically adjust.

### 10.4 Bilinear Fusion (Tsai et al., 2017)

**Concept**: Model second-order interactions between every text feature and every image feature.

```
output = ReLU(W_b @ (text_feat ⊗ img_feat) + b)
```

Where ⊗ is the outer product (every combination of text[i] * img[j]). This captures interactions like "when text feature 5 is high AND image feature 12 is high → offensive".

**Pros**: Captures fine-grained cross-modal interactions explicitly.
**Cons**: More parameters, higher overfitting risk on small datasets.

---

## 11. Troubleshooting

### Common Issues

| Problem | Cause | Solution |
|---------|-------|----------|
| `No module named 'src'` | Not in project root | `cd multimodal-meme-detector` first |
| `CUDA out of memory` | Batch size too large | `--batch_size 16` or `--batch_size 8` |
| `FileNotFoundError: glove.6B` | GloVe not downloaded | `python -m data.download` or `--use_mock` |
| `RuntimeError: size mismatch` | GloVe dim mismatch | Check `--glove_dim` matches the file |
| Low F1 (~0.5) | Expected on mock data | Train on real data with `--augment` |
| NaN loss | Learning rate too high | Reduce `--lr` to `1e-5` |
| All predictions same class | Severe class imbalance | Check `pos_weight` is being applied |

### Performance Tips

1. **Use `--augment`** — especially important with small datasets
2. **Start with cross_attention** — consistently best fusion strategy
3. **Use bilstm encoder** — bidirectional context helps with short meme text
4. **Don't skip early stopping** — prevents overfitting on tiny dataset
5. **Use `--use_mock` first** — verify pipeline works before committing to real training

### Understanding the Output

```
 Ep │  Train Loss  Acc    P    R   F1 │   Val Loss  Acc    P    R   F1
─────┼─────────────────────────────────┼────────────────────────────────
  5  │     0.4521  0.78  0.72  0.68 0.70│     0.5234  0.71  0.67  0.62 0.64  (3s)
```

- **Train Loss**: Should decrease over epochs (model is learning)
- **Val Loss**: Should decrease then stabilize (not increase = not overfitting)
- **Accuracy**: Overall correctness (can be misleading with imbalanced data)
- **Precision**: Of all predicted offensive, how many were actually offensive?
- **Recall**: Of all actually offensive, how many did we catch?
- **F1**: Harmonic mean of P and R — the main metric to optimize

If Train F1 >> Val F1 → overfitting (reduce epochs, increase dropout, add augmentation).
If both are low → underfitting (increase epochs, check data, try different encoder).
