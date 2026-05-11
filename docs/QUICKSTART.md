# 🚀 Quick Start Guide (5 Minutes)

Get the offensive meme detector running in 5 steps. No GPU required for demo mode.

---

## Step 1: Clone & Install

```bash
git clone https://github.com/lexus-x/multimodal--meme-detector.git
cd multimodal-meme-detector
pip install -r requirements.txt
```

That's it for setup. ~2 minutes on a decent connection.

---

## Step 2: Run with Mock Data (No Download Needed)

The project includes a mock data generator so you can test the entire pipeline without downloading the real dataset:

```bash
# Train for 3 epochs on fake data — just to verify everything works
python -m src.train --use_mock --epochs 3 --batch_size 8
```

You should see output like:
```
Device: cpu
  Mock dataset: mock_data/ (60 samples)
  Loading GloVe ... using random embeddings (demo mode)
  Train: 42 | Val: 9
  Parameters: 138,234,561 total | 123,517,121 trainable | 14,717,440 frozen

 Ep │  Train Loss  Acc    P    R   F1 │   Val Loss  Acc    P    R   F1
─────┼─────────────────────────────────┼────────────────────────────────
  1  │     0.7123  0.52  0.48  0.55 0.51│     0.6934  0.56  0.50  0.60 0.55  (2s)
  2  │     0.6891  0.57  0.53  0.58 0.55│     0.6812  0.56  0.50  0.55 0.52  (2s)
  3  │     0.6654  0.60  0.56  0.62 0.59│     0.6756  0.56  0.52  0.58 0.55  (2s)
```

If you see this → everything works. The numbers will be random (it's fake data) — that's expected.

---

## Step 3: Launch the Live UI

```bash
python -m src.demo
```

Opens at **http://localhost:7860**. You can:
- Upload any image + type text → get a prediction
- Compare all loaded models side-by-side
- Read the architecture explanation

> **Note:** Without trained checkpoints, it loads demo models with random weights. The UI works — predictions just won't be meaningful until you train on real data.

---

## Step 4: Train on Real Data (Optional)

If you want actual results:

```bash
# 1. Download the dataset + GloVe embeddings
python -m data.download

# 2. Train with the best fusion strategy
python -m src.train \
    --fusion cross_attention \
    --text_encoder bilstm \
    --epochs 20 \
    --batch_size 32 \
    --lr 0.0001 \
    --augment
```

This will:
- Download GloVe (800MB, one-time) and the MultiOFF dataset
- Train for ~20 epochs (~10-30 min depending on hardware)
- Save the best model to `checkpoints/cross_attention_bilstm_*/best_model.pth`

---

## Step 5: Evaluate & Analyze

```bash
# Evaluate on test set
python -m src.evaluate --checkpoint checkpoints/cross_attention_bilstm_*/best_model.pth

# Generate comparison plots
python -m src.analyze --runs checkpoints/*/
```

---

## What You Get

| File | What it does |
|------|-------------|
| `src/train.py` | Train any combination of fusion + encoder |
| `src/evaluate.py` | Test set metrics + error analysis |
| `src/demo.py` | Interactive web UI |
| `src/analyze.py` | Publication-quality plots |
| `run_experiments.py` | Train all models + compare in one command |

---

## One-Liner Commands

```bash
# Smoke test (mock data, no GPU)
python -m src.train --use_mock --epochs 3

# Train best model
python -m src.train --fusion cross_attention --epochs 20 --augment

# Compare all fusion strategies
python run_experiments.py --epochs 10

# Launch UI
python -m src.demo --share  # creates public link

# Full evaluation
python -m src.evaluate --checkpoint checkpoints/cross_attention_bilstm_*/best_model.pth
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: gradio` | `pip install gradio` |
| `CUDA out of memory` | Reduce `--batch_size 16` or `--batch_size 8` |
| `No module named src` | Run from the project root directory |
| GloVe download fails | Download manually from http://nlp.stanford.edu/data/glove.6B.zip, extract to `glove.6B/` |
| No checkpoints found | Train first, or the UI will use random-weight demo models |
