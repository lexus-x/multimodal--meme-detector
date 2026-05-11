# Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        User Interface                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │  React App   │  │  Gradio Demo │  │  CLI / Scripts   │   │
│  │  (port 3000) │  │  (port 7860) │  │                  │   │
│  └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘   │
│         │                 │                    │             │
│         └────────┬────────┴────────────────────┘             │
│                  ▼                                           │
│         ┌────────────────┐                                  │
│         │  FastAPI Backend│  ← REST API + static serving     │
│         │  (port 8000)   │                                  │
│         └────────┬───────┘                                  │
└──────────────────┼──────────────────────────────────────────┘
                   ▼
         ┌─────────────────────┐
         │   ML Pipeline        │
         │                      │
         │  ┌────────────────┐  │
         │  │ Text Branch    │  │  GloVe → BiLSTM/CNN → 256d
         │  └───────┬────────┘  │
         │          │           │
         │  ┌───────┴────────┐  │
         │  │ Image Branch   │  │  VGG16/ResNet50 → FC → 256d
         │  └───────┬────────┘  │
         │          │           │
         │  ┌───────┴────────┐  │
         │  │ Fusion Layer   │  │  Early/Attention/Gated/Bilinear
         │  └───────┬────────┘  │
         │          │           │
         │  ┌───────┴────────┐  │
         │  │ Classifier     │  │  MLP 512→256→128→1
         │  └────────────────┘  │
         └─────────────────────┘
```

## Fusion Strategies

### Early Fusion (Baseline)
Simple concatenation of text and image features. Forces the MLP to learn all cross-modal interactions.

### Cross-Modal Attention (Recommended)
Text queries attend to image key/values via multi-head attention. Gives the model a structural prior for selective attention across modalities.

### Gated Fusion
Learned sigmoid gate dynamically weights each modality. Gate = σ(W[text;img]). Output = gate·text + (1-gate)·img.

### Bilinear Fusion
Second-order feature interactions via bilinear pooling. Captures multiplicative interactions between modalities.

## Data Flow

1. **Input**: Meme image (JPEG/PNG) + text string
2. **Preprocessing**: Image → 224×224 RGB tensor (ImageNet normalization); Text → tokenized indices (GloVe vocab)
3. **Feature Extraction**: Text → 256d vector; Image → 256d vector
4. **Fusion**: Combined into single feature vector (strategy-dependent dimensionality)
5. **Classification**: MLP → sigmoid → P(offensive)
6. **Output**: Label (Offensive/Non-offensive) + confidence score

## Training Details

- **Loss**: BCEWithLogitsLoss with pos_weight (handles class imbalance)
- **Optimizer**: Adam (lr=1e-4, weight_decay=1e-5)
- **Scheduler**: Cosine annealing
- **Regularization**: Dropout, gradient clipping (max_norm=1.0), data augmentation
- **Early stopping**: Patience of 5-7 epochs on validation F1
